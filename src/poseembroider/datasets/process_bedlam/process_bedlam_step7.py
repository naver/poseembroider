##############################################################
## poseembroider                                            ##
## Copyright (c) 2024                                       ##
## Institut de Robotica i Informatica Industrial, CSIC-UPC  ##
## and Naver Corporation                                    ##
## Licensed under the CC BY-NC-SA 4.0 license.              ##
## See project root for license details.                    ##
##############################################################

import os
import time
from tqdm import tqdm
import numpy as np
import torch

import poseembroider.config as config
import poseembroider.utils as utils

import text2pose.posefix.correcting as correcting


# UTILS
################################################################################

def get_annot_file(split, suffix):
	annot_file = os.path.join(config.BEDLAM_PROCESSED_DIR, f"bedlam_{split}.pkl")
	if suffix is not None and isinstance(suffix,str):
		annot_file = annot_file.replace('.pkl', f"_{suffix}.pkl")
	return annot_file


def load_selected_hidx(split, suffix):
	filepath = os.path.join(config.BEDLAM_PROCESSED_DIR, f"imgpaths_bedlam_to_selected_human_index_{split}.pkl")
	if suffix is not None and isinstance(suffix,str):
		filepath = filepath.replace('.pkl', f"_{suffix}.pkl")
	return utils.read_pickle(filepath)


def load_pair_info(split, kind, farther_sampling_size, suffix):
	filepath = os.path.join(config.BEDLAM_PROCESSED_DIR, f"bedlam_fix_pairs_{split}_{kind}-sequence_fs{farther_sampling_size}.pt")
	if suffix is not None and isinstance(suffix,str):
		filepath = filepath.replace('.pt', f"_{suffix}.pt")
	data = torch.load(filepath)
	# NOTE: pairs below are given in terms of local pose ids (pose B, pose A)
	return data['pairs'], data['local2global_pose_ids']


def merge_dicts(d1, d2):
	"""
	d1: {} or {k:list}
	d2: {k:str}
	"""
	if not d1: return {k:[v] for k,v in d2.items()} # case where d1 = {}
	return {k:v+[d2[k]] for k,v in d1.items()}
	

def instruct_bedlam(split, suffix, kind, farther_sample_size, nb_caps=3):
	"""
	Create a pickle file with the following format:
	{
		<pair_index>: ["instruction 1", ... "instruction `nb_caps`"],
		...
	}
	"""

	# load cache data
	annot_file = get_annot_file(split, suffix)
	annots = utils.read_pickle(annot_file)
	all_img_paths = sorted(list(annots.keys())) # (sorted)
	selected_hidx = load_selected_hidx(split, suffix) # (sorted by construction)
	_ = selected_hidx.pop("criteria")
	print("Done loading the dataset!")

	# build look up table: pose_id -> (image_path, hidx)
	id_2_refs, data_id = {}, 0
	for img_path in tqdm(all_img_paths): # (sorted order)
		for h_ind, h_idx in enumerate(selected_hidx[img_path]): # (sorted order by construction)
			id_2_refs[data_id] = (img_path, h_idx)
			data_id += 1
	print(f"Done building the pose look-up table: parsed {data_id} poses.")

	# load pairs (local pose ids)
	pairs_locposid, loc2glob_poseid = load_pair_info(split, kind, farther_sample_size, suffix)
	pairs_locposid = pairs_locposid[:,[1,0]] # (pose B, pose A) --> (pose A, pose B)
	print(f"Loaded {len(pairs_locposid)} pairs.")

	# gather coordinates and joint_rotations
	# consider only poses from the set contributing to the creation of pairs
	n_poses = len(loc2glob_poseid)
	coords = torch.zeros(n_poses, 52, 3)
	joint_rotations = torch.zeros(n_poses, 52, 3)
	for i, global_pose_id in enumerate(loc2glob_poseid):
		img_path, h_i = id_2_refs[global_pose_id]
		# coordinates
		coords_ = annots[img_path][h_i]["smplx_pose3d"] # canonical pose orientation
		coords_ = torch.from_numpy(coords_)
		coords_ = torch.cat([coords_[:22], coords_[25:25+30]]) # shape (nb joints, 3)
		coords[i] = coords_
		# joint rotations
		joint_rotations[i] = torch.from_numpy(np.concatenate([
						annots[img_path][h_i]['smplx_global_orient'],
						annots[img_path][h_i]["smplx_body_pose"],
						annots[img_path][h_i]["smplx_left_hand_pose"],
						annots[img_path][h_i]["smplx_right_hand_pose"]
					])).to(torch.float32).view(-1, 3) # (nb joints, 3)
	print(f"Gathered coordinates & joint rotations for the {n_poses} needed poses.")

	# get automatic modifiers
	t1 = time.time()
	print("Now captioning...")
	instructions = {}
	for i in range(nb_caps):

		print(f'\n### [{round(time.time()-t1)}] Processing caption {i+1}/{nb_caps}...')
		# define different kinds of instructions
		simplified_instructions = False
		if i == 1:
			simplified_instructions = True

		inst_i = correcting.main(pairs_locposid,
						   		coords,
								global_rotation_change=None,
								joint_rotations=joint_rotations,
								joint_rotations_type="smplx",
								load_contact_codes_file=(f"{config.BEDLAM_PROCESSED_DIR}/tmp_contact_codes.pt", True if (i>0) else False),
								random_skip=True,
								add_description_text_pieces=False,
								simplified_instructions=simplified_instructions,
								verbose=True,
								save_dir=False,
								ret_type="dict") # don't save intermediate files

		utils.write_pickle(inst_i, f'{config.BEDLAM_PROCESSED_DIR}/tmp_inst_{i}.pkl')

		instructions = merge_dicts(instructions, inst_i)
	print(f"Pair captioning process took {time.time() - t1} seconds.")

	# Saving
	saving_file = os.path.join(config.BEDLAM_PROCESSED_DIR, f"bedlam_{split}_{kind}-sequence_fs{farther_sample_size}_{nb_caps}mod{'_'+suffix if suffix else ''}.pkl")
	print(f"Saving into {saving_file}")
	utils.write_pickle(instructions, saving_file)


# MAIN
################################################################################

if __name__ == "__main__":

	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument('--split', type=str, choices=('training', 'validation'), default='validation')
	parser.add_argument('--kind', type=str, choices=('in', 'out'), help='defines the set of considered pairs: in- or out-of- sequence pairs')
	parser.add_argument('--farther_sample', type=int, default=0, help='defines the set of considered poses')
	args = parser.parse_args()
	
	suffix = "try"
	instruct_bedlam(args.split, suffix, args.kind, args.farther_sample)