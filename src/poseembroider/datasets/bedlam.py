##############################################################
## poseembroider                                            ##
## Copyright (c) 2024                                       ##
## Institut de Robotica i Informatica Industrial, CSIC-UPC  ##
## and Naver Corporation                                    ##
## Licensed under the CC BY-NC-SA 4.0 license.              ##
## See project root for license details.                    ##
##############################################################

import os
import glob
import torch
import numpy as np
import roma
from tqdm import tqdm
from PIL import Image

import poseembroider.config as config
import poseembroider.utils as utils
from poseembroider.datasets.base_dataset import get_scaled_bbox


class BEDLAM():
	"""
	Needs to be combined & inherited to fill some attributes such as
	self.tokenizer_name, self.images, self.poses etc...
	"""
	def __init__(self, split):
		self.image_dir = os.path.join(config.BEDLAM_IMG_DIR, self._convert_split_name(split))

	def _convert_split_name(self, split):
		return {'train':'training', 'val':'validation'}[split]


	# --------------------------------------------------------------------------
	# CACHE CREATION
	# --------------------------------------------------------------------------

	def _cache_exists(self,):
		return os.path.isfile(self.cache_file.format("images")) and \
			os.path.isfile(self.cache_file.format("poses")) and \
			(self.tokenizer_name is None or os.path.isfile(self.cache_file.format(f"tokenizedtexts_tokenizer_{self.tokenizer_name}")))


	def _create_cache(self):
		# create directory for caches if it does not yet exist
		cache_dir = os.path.dirname(self.cache_file)
		if not os.path.isdir(cache_dir):
			os.makedirs(cache_dir)
			print("Created directory:", cache_dir)
		# save caches
		utils.write_pickle({"criteria":self.criteria, "bboxes":self.bboxes, "images":self.images, "full_human_visibility":self.full_human_visibility, "camera_transformation":self.camera_transf}, self.cache_file.format("images"), tell=True)
		utils.write_pickle({"criteria":self.criteria, "poses":self.poses, "shapes":self.shapes}, self.cache_file.format("poses"), tell=True)
		if self.texts_raw is not ValueError:
			utils.write_pickle({"criteria":self.criteria, "texts":self.texts_raw}, self.cache_file.format("rawtexts"), tell=True)
		if self.tokenizer_name is not None:
			utils.write_pickle({"criteria":self.criteria, "texts_tokens":self.texts_tokens, "texts_length":self.texts_length, "tokenizer_name":self.tokenizer_name}, self.cache_file.format(f"tokenizedtexts_tokenizer_{self.tokenizer_name}"), tell=True)


	def _load_cache(self, load_raw_texts=False):
		d = utils.read_pickle(self.cache_file.format("images"))
		self.images, self.bboxes, self.full_human_visibility, self.camera_transf = d['images'], d['bboxes'], d['full_human_visibility'], d["camera_transformation"]
		d = utils.read_pickle(self.cache_file.format("poses"))
		self.poses = d['poses']
		self.shapes = d['shapes']
		if self.tokenizer_name is not None:
			d = utils.read_pickle(self.cache_file.format(f"tokenizedtexts_tokenizer_{self.tokenizer_name}"))
			self.texts_tokens, self.texts_length = d['texts_tokens'], d['texts_length']
		if load_raw_texts:
			self._load_raw_texts()
		self.criteria = d['criteria']

	
	def _load_raw_texts(self):
		# (specifically needed for text generation evaluation)
		d = utils.read_pickle(self.cache_file.format("rawtexts"))
		self.texts_raw = d['texts']


	def _load_farther_sampled_ids(self, reduced_set, split):
		"""
		reduced_set: number to which reduce the data, based on farther sampled
					 indices.
		"""
		if split not in ['training', 'validation']: split = self._convert_split_name(split)
		# load farther sampling file: get data indices to keep
		fs_file = os.path.join(config.BEDLAM_PROCESSED_DIR, f"farther_sample_{split}_%s_try.pt")
		chosen_size = max([a.split("_")[3] for a in glob.glob(fs_file % '*')])
		selected = torch.load(fs_file % chosen_size)[1]
		assert reduced_set < len(selected), f"Can't make a subset of {reduced_set} elements as only {len(selected)} elements were pre-selected."
		selected = selected[:reduced_set]
		return selected
	

	def _filter_out(self, keep_indices):
		"""filter out elements that are not in the selection list of indices"""
		# filtering function
		filter_inds = lambda x: {k:v for k,v in x.items() if k in keep_indices}
		# filter data dicts
		self.images = filter_inds(self.images)
		self.bboxes = filter_inds(self.bboxes)
		self.full_human_visibility = filter_inds(self.full_human_visibility)
		self.poses = filter_inds(self.poses)
		self.shapes = filter_inds(self.shapes)
		self.camera_transf = filter_inds(self.camera_transf)
		return filter_inds


	# --------------------------------------------------------------------------
	# DATA IMPORT
	# --------------------------------------------------------------------------

	def _load_data(self, reduced_set=False):
		"""
		Pick relevant information from the full caches to produce smaller caches.
		"""

		split_ = self._convert_split_name(self.split)
		flag_reduced_set = f"_fs{reduced_set}" if reduced_set else ""
		suffix = "_try"

		# subcache locations
		img_pose_cache_file_full = os.path.join(config.BEDLAM_PROCESSED_DIR, f"bedlam_{split_}{suffix}.pkl")
		if "overlap30_res224_j16_sf11" in self.version:
			human_selection_cache_file_full = os.path.join(config.BEDLAM_PROCESSED_DIR, f"imgpaths_bedlam_to_selected_human_index_{split_}{suffix}.pkl")
		else: raise NotImplementedError

		# load subcaches
		img_pose_data = utils.read_pickle(img_pose_cache_file_full)
		human_selection = utils.read_pickle(human_selection_cache_file_full)
		self.criteria = human_selection.pop("criteria")
		print("Humans chosen according to the following criteria:", self.criteria)
		all_img_paths = sorted(list(img_pose_data.keys()))

		# load farther sampling data ids if using a reduced set
		if reduced_set:
			selected_fs = self._load_farther_sampled_ids(reduced_set, self.split)
			consider = lambda data_id: data_id in selected_fs
		else:
			consider = lambda data_id: True	

		# build look up table: pose_id -> (image_path, hidx, h_ind)
		id_2_refs, data_id = {}, 0
		for img_path in tqdm(all_img_paths): # (sorted order)
			for h_ind, h_idx in enumerate(human_selection[img_path]): # (sorted order by construction)
				if consider(data_id): # limit construction to selected poses 
					id_2_refs[data_id] = (img_path, h_idx, h_ind)
				data_id += 1
		print(f"Done building the pose look-up table: parsed {data_id} poses, selected {len(id_2_refs)}.")

		# parse information
		self.images = {} # {data_id: image path}
		self.poses = {} # {data_id: 3D rotations for the body joints}
		self.shapes = {} # {data_id: body shape}
		self.bboxes = {} # {data_id: [x1,y1,x2,y2]}
		self.full_human_visibility = {} # {data_id: True|False}
		self.camera_transf = {} # {data_id: (R, t)} where 'R' is the rotation to apply on the body pelvis to orient it as in the image, and 't' is the projection translation
		for i, data_id in tqdm(enumerate(id_2_refs)):
			img_path, h_idx, h_ind = id_2_refs[data_id]
			self.images[data_id] = img_path
			self.full_human_visibility[data_id] = self._is_full_human_visible(img_pose_data[img_path][h_idx]["smplx_pose2d"])
			self.poses[data_id] = self._process_pose(img_pose_data[img_path][h_idx])
			self.shapes[data_id] = torch.from_numpy(img_pose_data[img_path][h_idx]["smplx_shape"]).to(torch.float32).view(-1)
			self.bboxes[data_id] = get_scaled_bbox(img_pose_data[img_path][h_idx]["smplx_tight_bbox"], scale_factor=self.criteria["scale_factor"])
			cam_rot = self._get_normalized_camera_rotation(
				image_orient=img_pose_data[img_path][h_idx]["smplx_root_cam"],
				normalized_orient=img_pose_data[img_path][h_idx]["smplx_global_orient"])
			self.camera_transf[data_id] = (cam_rot, img_pose_data[img_path][h_idx]["smplx_transl"])

		return split_, flag_reduced_set, suffix, id_2_refs


	# --------------------------------------------------------------------------
	# DATA PROCESSING
	# --------------------------------------------------------------------------

	def _process_pose(self, ann):
		# load pose rotations
		pose = torch.from_numpy(np.concatenate([
				ann['smplx_global_orient'],
				ann["smplx_body_pose"],
				ann["smplx_left_hand_pose"],
				ann["smplx_right_hand_pose"]
			])).to(torch.float32).view(-1, 3)
		# rotate the pose so it's in the same configuration as AMASS SMPL-H poses
		pose[0] = roma.rotvec_composition([torch.tensor([torch.pi/2, 0.0, 0.0]), pose[0]])
		return pose # shape (nb of joints, 3)


	def _is_full_human_visible(self, img_jts_coords, nb_considered_joints:int=22, img_maxx:int=1280, img_maxy:int=720):
		img_jts_coords = img_jts_coords[:nb_considered_joints]
		visible_jts_indices = ((0<img_jts_coords[:,0]) \
								* (img_jts_coords[:,0]<img_maxx) \
								* (0<img_jts_coords[:,1]) \
								* (img_jts_coords[:,1]<img_maxy)).astype(bool)
		return visible_jts_indices.sum() == nb_considered_joints


	def _get_normalized_camera_rotation(self, image_orient, normalized_orient):
		image_orient = torch.from_numpy(image_orient).view(1,3)
		normalized_orient = torch.from_numpy(normalized_orient).view(1,3)
		R = roma.rotvec_composition([image_orient, roma.rotvec_inverse(normalized_orient)])
		return R.view(3)


	# --------------------------------------------------------------------------
	# DATA LOADING
	# --------------------------------------------------------------------------

	def load_bedlam_image(self, image_path, bbox):
		image = Image.open(os.path.join(self.image_dir, image_path))
		# convert image
		if image.mode != 'RGB':
			image = image.convert('RGB')
		if 'closeup' in image_path:
			# PIL rotates counter-clockwise; we need to rotate it by 90° clockwise
			# using `expand=True` to keep the whole image
			image = image.rotate(270, expand=True)
		# crop image to the bounding box of the human
		image = image.crop(bbox)
		return image


	def get_bedlam_camera_rotation(self, cam_rot):
		"""
		Args:
			cam_rot: rotation modificator applied to the global pose orientation
					to reflect the view angle from the camera

		Returns:
			cam_rot: rotation modificator to apply on the *processed* global
				pose orientation to reflect the view angle from the camera, 
				*in a setting homogeneized accross datasets*.
		"""
		# NOTE: cam_rot was obtained on the "raw" (hip-normalized) global pose
		# orientation; but the global orientation of the pose handled by the
		# dataset was further modified ("processed") for cross-dataset
		# reference frame homogeneization: concretly, the global pose
		# orientation was further rotated by pi/2.
		# Now, to get the updated camera rotation modificator, we need to take
		# into account this homogeneization rotation `r1`, plus an added
		# homogeneization rotation `r2`, which is needed to fall back again in
		# the reference frame shared across dataset.
		r1 = torch.tensor([torch.pi/2, 0.0, 0.0])
		r2 = torch.tensor([-torch.pi/2, 0.0, 0.0])
		cam_rot = roma.rotvec_composition([r2, cam_rot, roma.rotvec_inverse(r1)])
		return cam_rot