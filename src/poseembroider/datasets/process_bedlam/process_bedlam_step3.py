##############################################################
## poseembroider                                            ##
## Copyright (c) 2024                                       ##
## Institut de Robotica i Informatica Industrial, CSIC-UPC  ##
## and Naver Corporation                                    ##
## Licensed under the CC BY-NC-SA 4.0 license.              ##
## See project root for license details.                    ##
##############################################################

# human selection
import os
from tqdm import tqdm
from tabulate import tabulate

import poseembroider.config as config
import poseembroider.utils as utils


# INPUT
################################################################################


import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--split', type=str, choices=('training', 'validation'), default='validation')
args = parser.parse_args()


# SETUP
################################################################################

# define selection criteria
criteria = dict(
	overlap_threshold = 0.3, # maximal percentage of occlusion by another human for a human that is more in the background to be considered (using the tight crops)
	min_number_of_visible_joints = 16, # minimum number of visible main body joints
	resolution_threshold = 224, # minimum resolution along one of the X/Y dimensions (using the scaled bbox)
	scale_factor = 1.1, # (margins) scale factor for the tight boxes considered to define the studied scaled image crops (these crops are further subject to the resolution constraint)
)

print("**Selection criteria:**")
print("-----------")
print(f"Tolerable inter-human overlap threshold (on tight crops) = {criteria['overlap_threshold']*100}%.")
print(f"Minimum number of visible joints (main body joints) = {criteria['min_number_of_visible_joints']}.")
print(f"Minimum resolution along one of the X/Y dimensions (on scaled crops) = {criteria['resolution_threshold']}.")
print(f"Scale factor for the tight boxes considered to define scaled crops = {criteria['scale_factor']}.\n")

# global stats
selection_stats = {k:0 for k in ['non-overlap', 'min-resolution', 'min-visibility']}


# UTILS
################################################################################

def load_annots(split, suffix):
	annot_file = os.path.join(config.BEDLAM_PROCESSED_DIR, f"bedlam_{split}.pkl")
	if suffix is not None and isinstance(suffix,str):
		annot_file = annot_file.replace('.pkl', f"_{suffix}.pkl")
	return utils.read_pickle(annot_file)

def count_humans(d):
	n_total_humans = 0
	for v in d.values():
		n_total_humans += len(v)
	return n_total_humans

def this_box_is_overlapped(this_box, other_box, threshold=0.4):
	# determine the (x, y)-coordinates of the intersection rectangle
	xA = max(this_box[0], other_box[0])
	yA = max(this_box[1], other_box[1])
	xB = min(this_box[2], other_box[2])
	yB = min(this_box[3], other_box[3])
	# compute the area of intersection rectangle
	inter_area = abs(max((xB - xA, 0)) * max((yB - yA), 0))
	if inter_area == 0:
		return 0
	# compute the area of this box
	this_box_area = abs((this_box[2] - this_box[0]) * (this_box[3] - this_box[1]))
	# compute the proportion of overlap
	prop_overlap = inter_area/this_box_area
	return prop_overlap > threshold

def get_scaled_bbox(corners, scale_factor=1.1, max_x=1280, max_y=720):
	x1,y1,x2,y2 = corners # tight bbox
	# using as offset for all margins a percentage of the diagonal
	offset = (scale_factor-1) * ((x2-x1)**2 + (y2-y1)**2)**0.5 / 2
	x1 -= offset
	x2 += offset
	y1 -= offset
	y2 += offset
	# sanitize
	x1,y1,x2,y2 = int(x1), int(y1), int(x2), int(y2)
	x1, x2 = max(x1, 0), min(x2, max_x)
	y1, y2 = max(y1, 0), min(y2, max_y)
	return [x1, y1, x2, y2]

def select_humans(annot,
				  overlap_threshold=0.3,
				  min_number_of_visible_joints=16,
				  resolution_threshold=224,
				  scale_factor=1.1,
				  img_maxx = 1280,
				  img_maxy = 720,
				  applied_criteria = ['non-overlap', 'min-visibility', 'min-resolution']
				):

	# init
	nb_humans = len(annot)
	translations_z = [annot[j]['smplx_transl'][0,2] for j in range(nb_humans)] # depth

	s = {k:[] for k in ['non-overlap', 'min-visibility', 'min-resolution']}

	# iterate over humans in the image: check if they match the different criterias
	for h_idx_i in range(nb_humans):        

		# (1) non overlap
		keep = True
		for h_idx_j in range(nb_humans):
			if h_idx_i == h_idx_j: continue
			# make the overlap test on the tight boxes
			overlap = this_box_is_overlapped(annot[h_idx_i]['smplx_tight_bbox'],
											 annot[h_idx_j]['smplx_tight_bbox'],
											 threshold=overlap_threshold)
			if overlap and translations_z[h_idx_i] > translations_z[h_idx_j]: # consider people in the foreground
				keep = False
			if not keep:
				break
		if keep:
			s["non-overlap"].append(h_idx_i)
			selection_stats["non-overlap"] += 1
		
		# (2) full human visibility (joint visibility)
		nb_considered_joints = 22
		img_jts_coords = annot[h_idx_i]['smplx_pose2d'][:nb_considered_joints]
		visible_jts_indices = ((0<img_jts_coords[:,0]) \
								* (img_jts_coords[:,0]<img_maxx) \
								* (0<img_jts_coords[:,1]) \
								* (img_jts_coords[:,1]<img_maxy)).astype(bool)
		keep = visible_jts_indices.sum() >= min_number_of_visible_joints
		if keep:
			s["min-visibility"].append(h_idx_i)
			selection_stats["min-visibility"] += 1

		# (3) resolution
		x1, y1, x2, y2 = get_scaled_bbox(annot[h_idx_i]['smplx_tight_bbox'], scale_factor=scale_factor)
		keep = max(y2-y1, x2-x1) > resolution_threshold
		if keep:
			s["min-resolution"].append(h_idx_i)
			selection_stats["min-resolution"] += 1

	# find human indices that comply to all selected criteria
	s_all = set(s[applied_criteria[0]]) # init
	for ac in applied_criteria[1:]: # intersect iteratively
		s_all = s_all.intersection(set(s[ac]))

	return s_all


# SELECT
################################################################################

suffix = "try"

# load data
annots = load_annots(args.split, suffix=suffix)
n_total_humans_init = count_humans(annots)

# find the subset of human indices matching all criteria in each image
selected = {} # {image path: list of human indices}
for img_path in tqdm(annots.keys()):
	selected_human_indices = select_humans(annots[img_path], **criteria)
	selected[img_path] = sorted(selected_human_indices)

# get stats
n_total_humans_final = count_humans(selected)
print(f'[{args.split}] split')
tab = [['initially', n_total_humans_init, '']] + [[k, v, round(v/n_total_humans_init * 100, 2)] for k,v in selection_stats.items()] + [['finally', n_total_humans_final, round(n_total_humans_final/n_total_humans_init * 100, 2)]]
print(tabulate(tab, headers=['', 'nb humans', '%']))

# insert criteria information 
selected["criteria"] = criteria # add information about the used criteria

# save
filesave = os.path.join(config.BEDLAM_PROCESSED_DIR, f"imgpaths_bedlam_to_selected_human_index_{args.split}{'_'+suffix if suffix else ''}.pkl")
if not os.path.isfile(filesave):
	utils.write_pickle(selected, filesave)
	print("Saved:", filesave)