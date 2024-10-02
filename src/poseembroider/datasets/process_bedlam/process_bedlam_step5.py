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
import time
import math
import numpy as np
import torch
from tqdm import tqdm

import text2pose.posescript.captioning as captioning

import poseembroider.config as config
import poseembroider.utils as utils


# SETUP
################################################################################

# whether the body shape should play a role in the captioning pipeline
# (direct impact on contact codes; also beware that threshold measures for code
# categorization are not adapted to the body: they are kept the same for all
# body shapes)
CONSIDER_SHAPE = False


# UTILS
################################################################################

def get_annot_file(file_basename, split, suffix, ext="pkl"):
	annot_file = os.path.join(config.BEDLAM_PROCESSED_DIR, f"{file_basename}_{split}.{ext}")
	if suffix is not None and isinstance(suffix,str):
		annot_file = annot_file.replace(f'.{ext}', f"_{suffix}.{ext}")
	return annot_file


def load_farther_sampled_ids(split, farther_sample_size, suffix):
	filepath = get_annot_file("farther_sample", split+"_%s", suffix=(suffix or ""), ext="pt")
	chosen_size = max([a.split("_")[3] for a in glob.glob(filepath % '*')])
	selected = torch.load(filepath % chosen_size)[1]
	assert farther_sample_size < len(selected), f"Can't make a subset of {farther_sample_size} elements as only {len(selected)} elements were pre-selected."
	selected = selected[:farther_sample_size]
	return selected


def merge_dicts(d1, d2):
	"""
	d1: {} or {k:list}
	d2: {k:str}
	"""
	if not d1: return {k:[v] for k,v in d2.items()} # case where d1 = {}
	return {k:v+[d2[k]] for k,v in d1.items()}


def compute_coords(joint_rotations, shape_data=None):

	batch_size = 32
	
	# init body model
	body_model = utils.BodyModelSMPLX(batch_size=batch_size)
	body_model.eval()
	body_model.to('cpu')

	# init coords
	coords = torch.zeros(len(joint_rotations), 52, 3)
	joint_set = torch.concat([torch.arange(22), torch.arange(25,25+30)])
	
	# compute coords
	nb_iter = math.ceil(len(joint_rotations)/batch_size)
	for i in tqdm(range(nb_iter)):
		n_samples = min(len(joint_rotations[i*batch_size:]), batch_size)
		s = slice(i*batch_size, i*batch_size+n_samples)
		coords[s] = body_model(**utils.pose_data_as_dict(joint_rotations[s], code_base="smplx"), betas=shape_data[s]).joints[:,joint_set]

	return coords


def captionize_bedlam(split, farther_sample_size, suffix, nb_caps=3):
	"""
	Create a pickle file with the following format:
	{
		<img_path>: [
			["description 1 for human 1", ... "description `nb_caps` for human 1"],
			...
			["description 1 for human N", ... "description `nb_caps` for human N"]
		]
	}
	"""

	# load cache data
	annots = utils.read_pickle(get_annot_file("bedlam", split, suffix))
	all_img_paths = sorted(list(annots.keys())) # (sorted)
	selected_hidx = utils.read_pickle(get_annot_file("imgpaths_bedlam_to_selected_human_index", split, suffix)) # (sorted by construction)
	_ = selected_hidx.pop("criteria")
	if farther_sample_size>0:
		selected_fs = load_farther_sampled_ids(split, farther_sample_size, suffix) # ids valid in the nested for loops over all_img_paths then over selected_hidx
		consider = lambda data_id: data_id in selected_fs
		suffix = f'fs{farther_sample_size}_{suffix}'
	else:
		consider = lambda data_id: True
	print("Done loading the dataset!")

	# build look up table: pose_id -> (image_path, hidx)
	id_2_refs, data_id = {}, 0
	for img_path in tqdm(all_img_paths): # (sorted order)
		for h_ind, h_idx in enumerate(selected_hidx[img_path]): # (sorted order by construction)
			if consider(data_id): # limit construction to selected poses 
				id_2_refs[data_id] = (img_path, h_idx)
			data_id += 1
	print(f"Done building the pose look-up table: parsed {data_id} poses, selected {len(id_2_refs)}.")

	# gather coordinates and joint_rotations
	# consider only poses from the farther sampled set
	n_poses = len(id_2_refs)
	coords = None if CONSIDER_SHAPE else torch.zeros(n_poses, 52, 3)
	joint_rotations = torch.zeros(n_poses, 52, 3)
	shape_data = torch.zeros(n_poses, 11) if CONSIDER_SHAPE else None
	for i, data_id in enumerate(id_2_refs):
		img_path, h_i = id_2_refs[data_id]
		# joint rotations
		joint_rotations[i] = torch.from_numpy(np.concatenate([
						annots[img_path][h_i]['smplx_global_orient'],
						annots[img_path][h_i]["smplx_body_pose"],
						annots[img_path][h_i]["smplx_left_hand_pose"],
						annots[img_path][h_i]["smplx_right_hand_pose"]
					])).to(torch.float32).view(-1, 3) # (nb joints, 3)
		# coordinates (& shape)
		if CONSIDER_SHAPE:
			shape_data[i] = torch.from_numpy(annots[img_path][h_i]['smplx_shape']).to(torch.float32).view(-1)
			# NOTE: will fill the coords variable later
		else:
			coords_ = annots[img_path][h_i]["smplx_pose3d"] # canonical pose orientation
			coords_ = torch.from_numpy(coords_)
			coords_ = torch.cat([coords_[:22], coords_[25:25+30]]) # shape (nb joints, 3)
			coords[i] = coords_
	print(f"Gathered coordinates & joint rotations for the {n_poses} needed poses.")

	# recompute the coordinates with the body shape, if needed
	if CONSIDER_SHAPE:
		coords_from_shape_file = get_annot_file("bedlam_coords_from_shapes", split, suffix, ext="pt")
		if os.path.isfile(coords_from_shape_file):
			coords = torch.load(coords_from_shape_file)
			print("Load shape-dependent coordinates!")
		else:
			coords = compute_coords(joint_rotations, shape_data)
			torch.save(coords, coords_from_shape_file)
			print("Compute & saved shape-dependent coordinates:", coords_from_shape_file)

	# captionize
	t1 = time.time()
	print("Now captioning...")
	captions = {}
	for i in range(nb_caps):
		
		print(f'\n### [{round(time.time()-t1)}] Processing caption {i+1}/{nb_caps}...')
		# define different kinds of captions
		simplified_captions = False
		apply_transrel_ripple_effect = False
		apply_stat_ripple_effect = False
		if i == 1:
			simplified_captions = True
		elif i == 2:
			apply_transrel_ripple_effect = True
			apply_stat_ripple_effect = True
		
		# captionize!
		# NOTE: compute the contact codes only once, during the first pass, save
		# the results and load them for future passes (contact codes are
		# deterministic anyway)
		caps_i = captioning.main(coords,
								joint_rotations=joint_rotations,
								shape_data=shape_data, 
								joint_rotations_type="smplx",
								load_contact_codes_file=(f"{config.BEDLAM_PROCESSED_DIR}/tmp_contact_codes.pt", True if (i>0) else False),
								random_skip=True,
								simplified_captions=simplified_captions,
								apply_transrel_ripple_effect=apply_transrel_ripple_effect,
								apply_stat_ripple_effect=apply_stat_ripple_effect,								
						  		verbose=True,
								save_dir=False,
								ret_type="dict") # don't save intermediate files
		
		utils.write_pickle(caps_i, f'{config.BEDLAM_PROCESSED_DIR}/tmp_caps_{i}.pkl')

		captions = merge_dicts(captions, caps_i)
	print(f"Pose captioning process took {round(time.time() - t1)} seconds.")

	# Saving
	saving_file = os.path.join(config.BEDLAM_PROCESSED_DIR, f"bedlam_{split}_{nb_caps}caps{'_wshape' if CONSIDER_SHAPE else ''}{'_'+suffix if suffix else ''}.pkl")
	print(f"Saving into {saving_file}")
	utils.write_pickle(captions, saving_file)


# MAIN
################################################################################

if __name__ == "__main__":

	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument('--split', type=str, choices=('training', 'validation'), default='validation')
	parser.add_argument('--farther_sample', type=int, default=0, help='whether to caption the set of farther sampled poses only; size of the set')
	
	args = parser.parse_args()
	
	suffix = "try"
	captionize_bedlam(args.split, args.farther_sample, suffix)