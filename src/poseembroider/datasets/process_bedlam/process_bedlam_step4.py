##############################################################
## poseembroider                                            ##
## Copyright (c) 2024                                       ##
## Institut de Robotica i Informatica Industrial, CSIC-UPC  ##
## and Naver Corporation                                    ##
## Licensed under the CC BY-NC-SA 4.0 license.              ##
## See project root for license details.                    ##
##############################################################

import os
import torch
from tqdm import tqdm

import poseembroider.utils as utils
import poseembroider.config as config


# UTILS
################################################################################

def get_annot_file(split, suffix):
	annot_file = os.path.join(config.BEDLAM_PROCESSED_DIR, f"bedlam_{split}.pkl")
	if suffix is not None and isinstance(suffix,str):
		annot_file = annot_file.replace('.pkl', f"_{suffix}.pkl")
	return utils.read_pickle(annot_file)


def load_selected_hidx(split, suffix):
	filepath = os.path.join(config.BEDLAM_PROCESSED_DIR, f"imgpaths_bedlam_to_selected_human_index_{split}.pkl")
	if suffix is not None and isinstance(suffix,str):
		filepath = filepath.replace('.pkl', f"_{suffix}.pkl")
	return utils.read_pickle(filepath)


def MPJE(coords_set, coords_one):
	"""
	Args:
		coords_set: tensor (B, nb joints, 3)
		coords_one: tensor (nb joints, 3)
	
	Returns:
		tensor (B) giving the mean per joint distance between each pose in the
		set and the provided pose.
	"""
	return torch.norm(coords_set - coords_one, dim=2).mean(1) # (B)


def farther_sampling_resume(data, selected, distance, distance_func, nb_to_select=10):
	"""
	Args:
		data (torch.tensor): size (number of poses, number of joints, 3)
			or (number of poses, number of features)
		selected (list): indices of the poses that were already selected
		distance (torch.tensor): size (number of poses), distance between each
			pose and the closest pose of the selected set
		distance_func (function): computes the distance between 2 poses
		nb_to_select (int): number of data points to select, in addition of the
			ones already selected

	Returns:
		selected (list): indices of the selected poses
		distance (torch.tensor): size (number of poses), distance between each
			pose and the closest pose of the selected set
	"""

	nb_to_select = min(data.size(0)-len(selected), nb_to_select)

	for _ in tqdm(range(0, nb_to_select)):
		distance_update = distance_func(data, data[selected[-1]])
		distance = torch.amin(torch.cat((distance.view(-1,1), distance_update.view(-1,1)), 1), dim=1)
		selected.append(torch.argmax(distance).item())
		
	return selected, distance


def get_diversity_pose_order(split, suffix, nb_select=10, seed=0, resume=False):
	
	# load data
	annots = get_annot_file(split, suffix)
	selected_hidx = load_selected_hidx(split, suffix)
	all_img_paths = sorted(list(annots.keys()))
	print("Done loading the dataset!")

	# load coords
	coords = {} # {data_id: 3D coords (main keypoints)}
	references = {} # {data_id: [img_path, h_idxs]}
	data_id = 0
	for img_path in all_img_paths: # (sorted order)
		for h_idx in selected_hidx[img_path]: # (sorted order)
			coords[data_id] = torch.from_numpy(annots[img_path][h_idx]["smplx_pose3d"][:22])
			references[data_id] = [img_path, h_idx]
			data_id += 1
	
	# prepare data for the farther sampling
	# (resume, if applicable)
	coords = torch.stack([coords[did] for did in range(data_id)]) # (nb poses, nb joints, 3)
	nb_select = min(len(coords), nb_select)
	
	if resume:
		filepath_resume_from = os.path.join(config.BEDLAM_PROCESSED_DIR, f"farther_sample_{resume}_{suffix}.pt")
		references_, selected, distance = torch.load(filepath_resume_from)
		assert references == references_, "Cannot resume. Data changed!"
		print("Resuming from:", filepath_resume_from)
	else:
		selected = [seed] # to make the results somewhat reproducible
		distance = torch.ones(len(coords)) * float('inf')
		print(f"Farther sampling from seed {seed}.")
	
	# farther sample
	print(f"Sampling from {len(coords)} elements. Number of elements already selected: {len(selected)}.")
	selected, distance = farther_sampling_resume(coords, selected, distance, MPJE, nb_select)	

	# save
	filesave = os.path.join(config.BEDLAM_PROCESSED_DIR, f"farther_sample_{split}_{nb_select}_{suffix}.pt")
	torch.save([references, selected, distance], filesave)
	print("Saved:", filesave)


# MAIN
################################################################################

if __name__ == "__main__":

	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument('--split', type=str, choices=('training', 'validation'), default='validation')
	parser.add_argument('--nb_select', type=int, default=50000)
	parser.add_argument('--resume', type=int, default=0)
	args = parser.parse_args()
	
	suffix = "try"
	get_diversity_pose_order(split=args.split, suffix=suffix, nb_select=args.nb_select, resume=args.resume)