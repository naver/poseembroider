import os
import numpy as np
import torch
import smplx
from tqdm import tqdm
import pickle

os.umask(0x0002) # give write right to the team for all created files

import poseembroider.config as config
INPUT_DIR = os.path.join(config.BEDLAM_PROCESSED_DIR, "processed_npz")
OUTPUT_DIR = config.BEDLAM_PROCESSED_DIR


################################################################################
# UTILS
################################################################################

def get_body_model():
	body_model_smplx = smplx.create(config.SMPLX_BODY_MODEL_PATH,
				'smplx',
				gender='neutral',
				num_betas=11,
				use_pca = False,
				flat_hand_mean = True,
				batch_size=1)
	body_model_smplx.eval()
	body_model_smplx.to('cpu')
	return body_model_smplx

def pose_to_dict(data):
	d = dict(
		global_orient=data[:3],
		body_pose=data[3:66],
		jaw_pose=data[66:69],
		leye_pose=data[69:72],
		reye_pose=data[72:75],
		left_hand_pose=data[75:120],
		right_hand_pose=data[120:165],
	)
	return {k:v.reshape(-1,3) for k,v in d.items()}

def np_to_torch_bm_input(d):
	return {k:torch.from_numpy(v).reshape(1,-1) for k,v in d.items()}


def build_pkl(dir_to_npz, image_dir, annot_file, closeup=False):

	# Init
	# data holder
	imagename2annot = {}
	# list of data files
	fns = os.listdir(dir_to_npz)
	fns.sort()

	# if required, keep only the closeup files
	if closeup:
		_fns = []
		for x in fns:
			if 'closeup' in x:
				_fns.append(x)
		del fns
		fns = _fns

	# Loop
	nb_image = 0
	for i_fn, fn in enumerate(tqdm(fns)):
		
		print(f"Load data... (i_fn={i_fn} --- fn={fn})")

		# Loading npz file (smplx)
		annot_x = np.load(os.path.join(dir_to_npz, fn))
		# parse content
		imgname_array = annot_x['imgname']
		pose_array = annot_x['base_poses']
		joints3d_can_array = annot_x['joints3d_can']
		root_world_array = annot_x["root_world"]
		root_cam_array = annot_x["root_cam"]
		tight_bboxes_array = annot_x["tight_bboxes"]
		transl_array = annot_x["trans_project"]
		shape_array = annot_x["shape"]
		keypoints2d_array = annot_x['gtkps']
		K_array = annot_x['cam_int']
		H_array = annot_x['cam_ext']
		identity_array = annot_x['sub']

		# closeup - need to rotate the image
		if 'closeup' in fn: # `fn` is part of the path to the image (see below)
			print(f"Nb image to rotate = {len(imgname_array)}")

		# Group persons per image
		imgname2ids = {}
		imgname2exist = {}

		img_dir = os.path.join(image_dir, fn.replace('_6fps.npz', '/png'))
		working_img_dir = img_dir
		if not os.path.isdir(working_img_dir):
			working_img_dir = img_dir.replace("/png", "_6fps/png")
		if not os.path.isdir(working_img_dir):
			working_img_dir = img_dir.replace("/png", "_30fps/png")

		for i, x in enumerate(imgname_array.tolist()):
			# checking if the image exists
			img_path = os.path.join(working_img_dir, x)
			if img_path in imgname2exist.keys():
				if not imgname2exist[img_path]:
					continue
			else:
				exist = os.path.exists(img_path)
				imgname2exist[img_path] = exist
				if not exist:
					print(f"{img_path} does not exists...")
					continue
			# appending the person_id to the image
			imgname2ids[x] = imgname2ids.get(x, []) + [i]
	
		# Iterate over images
		for imgname, idxs in imgname2ids.items():

			img_path = os.path.join(working_img_dir[len(image_dir)+1:], imgname_array[idxs[0]])
			persons = []

			for i in idxs:

				# get camera info
				K = K_array[i] # [3,3]
				focal = np.asarray([K[0,0], K[1,1]])
				princpt = np.asarray([K[0,-1], K[1,-1]])
				H = H_array[i] # [4,4]

				# get base pose
				d_pose = pose_to_dict(pose_array[i]) # with original global orientation

				# populate with values of interest
				person = {
					'princpt':princpt.astype(np.float32).reshape(2),
					'focal':focal.astype(np.float32).reshape(2),
					'identity': identity_array[i],
					'smplx_gender': 'neutral',
					'smplx_shape': shape_array[i].astype(np.float32).reshape(11),
					'smplx_tight_bbox': tight_bboxes_array[i].astype(np.float32),
					'smplx_transl': transl_array[i].astype(np.float32),
					'smplx_root_cam': root_cam_array[i].astype(np.float32),
					'smplx_root_world': root_world_array[i].astype(np.float32),
					'smplx_pose2d': keypoints2d_array[i][:,:2].astype(np.int32).reshape(127,2), # keypoint positions in the image (pixels) ; NOTE: removing the third column, which is always 1 (does not correspond to joint visibility, perhaps to confidence)
					'smplx_pose3d': joints3d_can_array[i].astype(np.float32).reshape(127,3), # canonical 3D keypoint positions (ie. neutral body, default shape)
				}
				person.update({f"smplx_{k}":v.astype(np.float32) for k,v in d_pose.items()})  # axis-angle
				persons.append(person)

			# Append
			imagename2annot[img_path] = persons
			nb_image +=1

	# Saving
	print(f"Saving into {annot_file}")
	with open(annot_file, 'wb') as f:
		pickle.dump(imagename2annot, f, protocol=pickle.HIGHEST_PROTOCOL)

	return imagename2annot


################################################################################
# DATASET
################################################################################

def get_annot_file(split, suffix=None, annot_file=None):

	#### GET FILE DATA
	# split name normalization
	closeup = 'closeup' in split
	if 'training' in split:
		split = 'training'

	# cache file location
	if annot_file:
		annot_file = annot_file
	else:
		annot_file = os.path.join(OUTPUT_DIR, f"bedlam_{split}.pkl")
	if suffix is not None and isinstance(suffix,str):
		annot_file = annot_file.replace('.pkl', f"_{suffix}.pkl")
	print("Expecting data at:", annot_file)

	return annot_file, closeup, split


class BEDLAM_ORGANIZER():

	name = 'bedlam'

	def __init__(self,
				 # Main args
				 split='training',
				 image_dir=config.BEDLAM_IMG_DIR,
				 force_build_dataset=False,
				 suffix=None,
				 annot_file=None,
				 ):
		assert split in ['training', 'validation', 'training_closeup']

		# define location of cache file (self.annot_file)
		self.annot_file, self.closeup, self.split = get_annot_file(split, suffix, annot_file)

		# setup
		self.image_dir = os.path.join(image_dir, f"{self.split}")

		# cache data or load cache data
		self.annots = None
		if force_build_dataset or not os.path.isfile(self.annot_file):
			self.annots = self.build_dataset()
			print("Done building the dataset!")
		if self.annots is None: # there exist a cache, and we are allowed to use it
			with open(self.annot_file, 'rb') as f:
				self.annots = pickle.load(f)
			print("Done loading the dataset!")
		self.imagenames = list(self.annots.keys())
		self.imagenames.sort()
		
		# display information
		self.check_images_exist() # performed only if `check_images=True` as argument
		if self.split != 'test':
			self.compute_stats()

	def build_dataset(self):
		dir_to_npz = os.path.join(INPUT_DIR, self.split)
		image_dir = os.path.join(config.BEDLAM_IMG_DIR, self.split)
		return build_pkl(dir_to_npz, image_dir, self.annot_file, self.closeup)

	def compute_stats(self):
		nb_person = 0
		for key, values in self.annots.items():
			nb_person += len(values)
		print(f"*** {len(self.imagenames)} images - {nb_person} humans ({nb_person/len(self.imagenames):.2f}/img) ***")


################################################################################
# MAIN
################################################################################

if __name__ == "__main__":

	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument('--split', type=str, choices=('training', 'validation'), default='validation')
	args = parser.parse_args()

	force_build_dataset = True
	
	dataset = BEDLAM_ORGANIZER(split=args.split, force_build_dataset=force_build_dataset, suffix="try")