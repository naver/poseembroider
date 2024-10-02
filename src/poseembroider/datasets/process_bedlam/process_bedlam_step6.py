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
from tqdm import tqdm
import random
import numpy as np
import torch

import poseembroider.config as config
import poseembroider.utils as utils


# SETUP
################################################################################

# dataset
dataset_version = "bedlamscript-overlap30_res224_j16_sf11"

# model for posetext features
posetext_shortname = "posetext_model_bedlamscript"
shortname_2_model_path = utils.read_json("shortname_2_model_path.json")
posetext_model_for_features = os.path.join(config.GENERAL_EXP_OUTPUT_DIR, shortname_2_model_path[posetext_shortname])

# selection constraint
MAX_TIMEDIFF = 0.5
FT_TOP = 100
MIN_SIM = 0.7
MAX_SIM = 0.9
PC_CONSTRAINT_WINDOW = 3
PC_CONSTRAINT_KEEP = 3 # must be < PC_CONSTRAINT_WINDOW; ==> prevent role unicity but limits the high multiplicity of the role pose "A"
MIN_PCDIFF = {"in-sequence":15, "out-sequence":20}


# UTILS - create & load intermediate files 
################################################################################

def get_annot_file(split, suffix):
	annot_file = os.path.join(config.BEDLAM_PROCESSED_DIR, f"bedlam_{split}.pkl")
	if suffix is not None and isinstance(suffix,str):
		annot_file = annot_file.replace('.pkl', f"_{suffix}.pkl")
	return annot_file


def load_selected_hidx(split, suffix):
	assert dataset_version == "bedlamscript-overlap30_res224_j16_sf11", "Conflict between data loaded now, and data used to eg. infer posetext features."
	filepath = os.path.join(config.BEDLAM_PROCESSED_DIR, f"imgpaths_bedlam_to_selected_human_index_{split}.pkl")
	if suffix is not None and isinstance(suffix,str):
		filepath = filepath.replace('.pkl', f"_{suffix}.pkl")
	return utils.read_pickle(filepath)


def load_farther_sampled_ids(split, farther_sample_size, suffix):
	filepath = os.path.join(config.BEDLAM_PROCESSED_DIR, f"farther_sample_{split}_%s.pt")
	if suffix is not None and isinstance(suffix,str):
		filepath = filepath.replace('.pt', f"_{suffix}.pt")
	chosen_size = max([a.split("_")[3] for a in glob.glob(filepath % '*')])
	selected = torch.load(filepath % chosen_size)[1]
	assert farther_sample_size < len(selected), f"Can't make a subset of {farther_sample_size} elements as only {len(selected)} elements were pre-selected."
	selected = selected[:farther_sample_size]
	return selected


def get_time_difference_mat(split, id_2_refs, suffix):
	time_diff_mat_filepath = os.path.join(config.BEDLAM_PROCESSED_DIR, f"time_difference_{split}{'_'+suffix if suffix else ''}.pt")
	if not os.path.isfile(time_diff_mat_filepath):
		N = len(id_2_refs)
		time_diff_mat = torch.zeros(N, N, device="cpu")
		imgname2frameind = lambda img_name: int(img_name.split("_")[-1][:-4]) # [:-4] to remove ".png"
		for i1, id1 in tqdm(enumerate(id_2_refs.values())):
			for i2, id2 in enumerate(id_2_refs.values()):
				# NOTE: image_path is like: .../20221010_3_1000_batch01hand_6fps/png/seq_000000/seq_000000_0010.png
				imgp1, _, subj1 = id1
				imgp2, _, subj2 = id2
				if i1!=i2: # proceed, if not processing the pose with itself
					# if it's the same motion sequence (ie. same video and same person)
					if subj1 == subj2 and os.path.dirname(imgp1) == os.path.dirname(imgp2):
						time_diff_mat[i1,i2] = imgname2frameind(imgp1) - imgname2frameind(imgp2) # difference in frame position
		# normalize by the framerate to get time difference
		# NOTE: BEDLAM sequences were all registered in 30fps.
		# The frame difference computation is the same for the "6fps" sequences
		# (it's just that less images were kept, but their names follow the 30fps schema)
		time_diff_mat /= 30.0
		torch.save(time_diff_mat, time_diff_mat_filepath)
		print("Saved", time_diff_mat_filepath)
	else:
		time_diff_mat = torch.load(time_diff_mat_filepath)
		print("Loaded", time_diff_mat_filepath)
	return time_diff_mat


def get_posesecript_features(split, farther_sample_size, suffix):
	# NOTE: the order of the produced pose features is fine, because the look up
	# table `id_2_refs` has the same ordering as the data underlying the BEDLAMScript
	# dataset. IMPORTANT!
	retrieval_ft_filepath = os.path.join(config.BEDLAM_PROCESSED_DIR, f"retrieval_features_{split}_{posetext_shortname}{'_'+suffix if suffix else ''}.pt")
	if not os.path.isfile(retrieval_ft_filepath):
		from poseembroider.datasets.bedlam_script import BEDLAMScript
		from poseembroider.assisting_models.poseText.utils_poseText import load_model
		device = torch.device('cuda:0')
		batch_size = 32
		model, _ = load_model(posetext_model_for_features, device)
		# create dataset
		split_ = {"training":"train", "validation":"val"}[split]
		dataset = BEDLAMScript(version=dataset_version, split=split_, num_body_joints=model.pose_encoder.num_body_joints, reduced_set=farther_sample_size, item_format='p')
		data_loader = torch.utils.data.DataLoader(
			dataset, sampler=None, shuffle=False,
			batch_size=batch_size,
			num_workers=8,
			pin_memory=True,
			drop_last=False
		)
		# compute pose features
		poses_features = torch.zeros(len(dataset), model.latentD).to(device)
		for i, batch in tqdm(enumerate(data_loader)):
			poses = batch['poses'].to(device)
			with torch.inference_mode():
				pfeat = model.pose_encoder(poses)
				poses_features[i*batch_size:i*batch_size+len(poses)] = pfeat
		poses_features /= torch.linalg.norm(poses_features, axis=1).view(-1,1)
		poses_features = poses_features.cpu()
		torch.save(poses_features, retrieval_ft_filepath)
		print("Saved", retrieval_ft_filepath)
	else:
		poses_features = torch.load(retrieval_ft_filepath)
		print("Loaded", retrieval_ft_filepath)
	return poses_features


def get_posecodes(split, id_2_refs, annots, suffix):
	posecodes_filepath = os.path.join(config.BEDLAM_PROCESSED_DIR, f"posecodes_{split}{'_'+suffix if suffix else ''}.pt")
	if not os.path.isfile(posecodes_filepath):
		from text2pose.posescript.utils import prepare_input
		from text2pose.posescript.captioning import prepare_posecode_queries, prepare_super_posecode_queries, infer_posecodes, POSECODE_INTPTT_NAME2ID
		# format joint coordinates for input (matrix)
		coords = torch.zeros(len(id_2_refs), 52, 3)
		for i, refs in enumerate(id_2_refs.values()):
			img_path, h_i, _ = refs
			coords_ = annots[img_path][h_i]["smplx_pose3d"] # canonical pose orientation
			coords_ = torch.from_numpy(coords_)
			coords_ = torch.cat([coords_[:22], coords_[25:25+30]]) # shape (nb joints, 3)
			coords[i] = coords_
		print(f"Considering {len(coords)} pose coordinates.")
		# Select & complete joint coordinates (prosthesis phalanxes, virtual joints)
		coords = prepare_input(coords)
		# Prepare posecode queries (hold all info about posecodes, essentially using ids)
		p_queries = prepare_posecode_queries()
		sp_queries = prepare_super_posecode_queries(p_queries)
		# Eval & interprete & elect eligible elementary posecodes
		p_interpretations, p_eligibility = infer_posecodes(coords, p_queries, sp_queries, verbose=True)
		saved_filepath = os.path.join(config.BEDLAM_PROCESSED_DIR, f"posecodes_intptt_eligibility_{split}{'_'+suffix if suffix else ''}.pt")
		torch.save([p_interpretations, p_eligibility, POSECODE_INTPTT_NAME2ID], saved_filepath)
		# Extract posecode data for constraint
		p_interpretations["superPosecodes"] = (p_eligibility["superPosecodes"] > 0).type(p_interpretations["angle"].dtype)
		del p_eligibility
		# Save
		torch.save(p_interpretations, posecodes_filepath)
		print("Saved", posecodes_filepath)
	else:
		p_interpretations = torch.load(posecodes_filepath)
		print("Loaded", posecodes_filepath)
	max_posecodes = sum([p.shape[1] for p in p_interpretations.values()])
	print("Number of posecodes:", max_posecodes)
	return p_interpretations, max_posecodes


# UTILS - enforce constraints 
################################################################################

def sequence_constraint(kind, time_diff_mat):

	if kind == "in-sequence":
		# consider pairs extracted from the same sequence (frame diff > 0),
		# in a forward (.abs()) short (< MAX_TIMEDIFF seconds) motion
		s = torch.logical_and(time_diff_mat>0, time_diff_mat.abs()<MAX_TIMEDIFF)
		return s, time_diff_mat
		
	elif kind == "out-sequence":
		# consider pairs of poses that do not belong to the same sequence
		# (remove pairs made of twice the same pose (ie. diagonal))
		s = torch.logical_and(time_diff_mat==0, 1 - torch.diag(torch.ones(len(time_diff_mat))))
		return s, None


def compute_feature_matrix(retrieval_features):
	# Compute cosine similarity matrix to compare poses.
	# The higher the score, the most similar the 2 poses.
	# Make one pose be orthogonal with itself, to prevent selection of pairs A --> A.
	ft_mat = torch.mm(retrieval_features, retrieval_features.t()).fill_diagonal_(0)
	return ft_mat, [MIN_SIM, MAX_SIM]


def feature_constraint(ft_mat, ft_thresholds):

	# Consider poses that are similar
	n_poses = ft_mat.shape[0]
	s = torch.zeros(n_poses, n_poses).scatter(1, torch.topk(ft_mat, k=FT_TOP)[1], True)
	# in the code line above:
	# torch.topk(mat, k=FT_TOP)[1] gives the indices of the top K values in mat (the top K is computed per row)
	# m.scatter(1, indices, values) set the provided values at the provided indices in the provided matrix m
	# so basically, we keep only the top K poses A that are the most similar to pose B
	# (where the rows are for poses B, and columns for poses A)

	# Additionally remove poses that are either really too similar or too different
	s = torch.logical_and(s, ft_mat>ft_thresholds[0]) # min threshold
	s = torch.logical_and(s, ft_mat<ft_thresholds[1]) # max threshold
	return s


def posecode_constraint(kind, s, p_inptt, ft_mat, max_posecodes):

	n_poses = s.shape[0]
	# store the number of different posecodes for each pair along with the rank
	# of pose A wrt pose B (obtain the first by taking the modulo and the second
	# by taking the division of the absolute value of `pc_info` with
	# `max_posecodes`; a rank of 0 means that the pose A was not selected for a
	# pair with pose B, the sign indicates whether there were more eligible
	# poses than PC_CONSTRAINT_KEEP (and thus A was really "selected") or not
	# (and thus, A was just one of the only choices (presumably yielding a pair
	# of lower quality)); for "in-sequence" pairs, rank will always be 1)
	pc_info = torch.zeros(n_poses, n_poses).type(torch.int)
	# group pairs according to pose B
	pairs = {p1:torch.masked_select(torch.arange(n_poses), s[p1]) for p1 in range(n_poses) if s[p1].sum()}
	nb_zeroed_B_poses = 0 # number of poses B for which no pose A can be found to satisfy the constraints

	# define posecode constraint
	for index, indices in tqdm(pairs.items()):
		# compute number of different posecodes
		pc_different = torch.ones(len(indices)).type(torch.int) * max_posecodes
		for pc_kind in p_inptt:
			pc_different -= (p_inptt[pc_kind][index] == p_inptt[pc_kind][indices]).sum(1)
		pc_info[index, indices] = pc_different # store number of different posecodes
		# apply condition constraint
		pc_constraint = (pc_different > MIN_PCDIFF[kind]).type(torch.bool)
		# For "out-of-sequence" pairs, apply further selection
		if kind == "out-sequence":
			# At this point, we need to update `s` based on which pairs made with
			# `index` (ie. indices) we decide to keep. Of course, we won't keep
			# indices[~constraint], but we may reject even more than only those: we
			# choose to first select only the best PC_CONSTRAINT_WINDOW among pose
			# pairs that satisfy the constraint (ie. indices[constraint]); where the
			# "best" are defined with regard to the feature similarity. Then we
			# randomly choose PC_CONSTRAINT_KEEP among the maximum
			# PC_CONSTRAINT_WINDOW poses available, to reduce the number of pairs
			# (==> maximum PC_CONSTRAINT_KEEP pairs with pose B as receiving pose)
			try:
				# first select the top PC_CONSTRAINT_WINDOW, and store information
				inds = torch.topk(ft_mat[index, indices[pc_constraint]], k=PC_CONSTRAINT_WINDOW)[1]
				chosen = indices[pc_constraint][inds] # get actual matrix indices
				pc_info[index, chosen] += (1 + torch.arange(PC_CONSTRAINT_WINDOW).int())*max_posecodes # store rank (+1 to distinguish from the '0' coeffs meaning the pose was not ranked)
				# then select randomly PC_CONSTRAINT_KEEP among those
				rchosen = torch.tensor(random.sample(range(len(chosen)), PC_CONSTRAINT_KEEP)).long()
				selected_by_this_constraint = torch.zeros(n_poses)
				selected_by_this_constraint[chosen[rchosen]] = True
				s[index] = torch.logical_and(s[index], selected_by_this_constraint)
			except RuntimeError:
				# Error occur when computing 'inds': selected index k out of
				# range (k<PC_CONSTRAINT_WINDOW)
				# ie. there are less than PC_CONSTRAINT_WINDOW poses A available
				# for this pose B, that satisfy the posecode constraint in
				# addition of the previous constraints. Thus, consider all
				# available poses A. 
				chosen = indices[pc_constraint]
				# store information
				pc_info[index, indices[pc_constraint]] += (1 + torch.arange(len(indices[pc_constraint])).int())*max_posecodes
				pc_info[index, indices[pc_constraint]] *= -1 # non-positive number to distinguish from actual selection (here, we take everything that is available ==> not necessarily high quality pairs)
				if len(chosen) == 0:
					nb_zeroed_B_poses += 1
				# keep at maximum PC_CONSTRAINT_KEEP pairs for each pose B
				rchosen = torch.tensor(random.sample(range(len(chosen)), min(PC_CONSTRAINT_KEEP, len(chosen)))).long()
				selected_by_this_constraint = torch.zeros(n_poses)
				selected_by_this_constraint[chosen[rchosen]] = True
				s[index] = torch.logical_and(s[index], selected_by_this_constraint)
		# For "in-sequence" pairs, the posecode constraint is just applied
		# to eliminate pairs that would not respect the constraint (there is no
		# further selection, as there are not already so many in-sequence pairs)
		elif kind == "in-sequence":
			s[index, indices[~pc_constraint]] = False
			pc_info[index, indices[pc_constraint]] += max_posecodes

	return s, pc_info, nb_zeroed_B_poses


def pose_unicity_in_role_for_insequence_mining(s, time_diff_mat):
	# Use distinct starting & receiving poses (ie. the same pose cannot be used
	# twice as pose A or pose B (for A --> B))
	pairs = torch.where(s) # available pairs
	for p in tqdm(range(len(pairs[0]))):
		# a) distinct receiving poses
		inds = torch.where(pairs[0] == pairs[0][p])[0].tolist() # look for all pairs using pose p as B
		# keep only the direct pair (ie. pair with the minimum time difference)
		keep = np.argmin([time_diff_mat[pairs[0][p], pairs[1][i]] for i in inds])
		for ii, i in enumerate(inds):
			if ii != keep:
				s[pairs[0][p], pairs[1][i]] = False
		# b) distinct starting poses
		inds = torch.where(pairs[1] == pairs[1][p])[0].tolist() # look for all pairs using pose p as A
		# keep only the direct pair (ie. pair with the minimum time difference)
		keep = np.argmin([time_diff_mat[pairs[0][i], pairs[1][p]] for i in inds])
		for ii, i in enumerate(inds):
			if ii != keep:
				s[pairs[0][i], pairs[1][p]] = False
	return s


# UTILS - ordering
################################################################################

def get_ordered_annotation_pairs(kind, s, id_2_refs):
	
	# Setup
	# create a dict to convert a global index (denoting the pose in "general")
	# to its corresponding index in the matrices, and vice-versa
	global2local = {pid:i for i, pid in enumerate(id_2_refs.keys())}
	local2global = list(id_2_refs.keys())

	# Get distinct poses B
	pairs = torch.stack(torch.where(s)).t()
	poses_B = torch.unique(pairs[:, 0]) # indices within the split ("local" indices)
	# get global indices of such poses
	poses_B_pids = torch.tensor(local2global)[poses_B] # ("global" indices)
	
	# Farther sampling
	# global indices reflect the farther sampling order, so no need to actually
	# farther sample the poses B, just rank the poses based on their global
	# ID to have them ordered in the farther sampling order
	poses_B_pids = list(torch.sort(poses_B_pids).values) # ordered!

	# Gather pairs based on pose B, following the farther sampling order
	# * in-sequence pairs: no game on the ordering, just consider the mined
	#   pairs as is
	# * out-of-sequence pairs: we are also interested in reverse relations;
	#   ie. pose B should be considered as pose A as well
	pairs_order = []
	only_one_way = 0
	b_considered = 0
	# create a new pose selection matrix, to include the pairs in reverse
	# direction B --> a without affecting the mining of pairs initially
	# selected `s` for each pose A (see below); only necessary for
	# out-of-sequence pair selection
	if kind == "in-sequence":
		s_new = s
	elif kind == "out-sequence":
		s_new = torch.zeros_like(s)
		A_weights = [1/aw if aw>0 else 0 for aw in s.sum(0)] # define sampling weights for poses A based on the number of times they could be used as pose A, globally (when considering all poses B at once)
	# iterate over poses B
	start_progression = len(poses_B_pids)
	while len(poses_B_pids): # poses_B_pids is ordered!
		B_pid = poses_B_pids[0]
		B_split_index = global2local[B_pid.item()]
		A_split_indices = torch.where(s[B_split_index])[0].tolist()
		if kind == "in-sequence":
			pairs_order += [[B_split_index, x] for x in A_split_indices]
		elif kind == "out-sequence":
			# whenever it is possible, consider both pairs x --> B and B --> x
			# NOTE: pair x --> B is stored as [B, x]
			X = [x for x in A_split_indices if local2global[x] in poses_B_pids] # ie. x is still available for the way back
			if X:
				x = random.choices(X, weights=[A_weights[x] for x in X])[0] # use higher sampling probability for poses A that are unique to this pose B, regarding the whole split; to optimize the number of two-way pairs
				# forward direction
				pairs_order += [[B_split_index, x]]
				# reverse direction
				pairs_order += [[x, B_split_index]]
				x_pid = local2global[x]
				poses_B_pids.remove(x_pid) # don't consider x for its own way forward anymore
			else:
				x = random.choices(A_split_indices, weights=[A_weights[x] for x in A_split_indices])[0] # use higher sampling probability for poses A that are unique to this pose B, regarding the whole split; to optimize the number of two-way pairs
				# only one way available: x --> B
				pairs_order += [[B_split_index, x]]
				only_one_way += 1
		progress = round((1 - len(poses_B_pids)/start_progression) * 100, 2)
		print(f"Progress (%): {progress}", end='\r', flush=True)
		poses_B_pids.remove(B_pid)
		b_considered += 1

	pairs_order = torch.tensor(pairs_order)
	if kind == "out-sequence":
		# rectify the pairing matrix
		s_new[pairs_order.t().unbind()] = True
		# provide information about one-way pairs
		print(f"Number of one-way pairs: {only_one_way} (among {b_considered} distinct poses B considered in turn for pairing).\nTotal number of pairs = 2 * (number of poses B considered in turn) - (one way pairs).")
	
	return pairs_order, s_new


# MAIN
################################################################################


def select_pairs(split, kind, farther_sample_size, suffix):
	
	# (1) SETUP ----------------------------------------------------------------

	# load cache data
	annot_file = get_annot_file(split, suffix)
	annots = utils.read_pickle(annot_file)
	all_img_paths = sorted(list(annots.keys())) # (sorted)
	selected_hidx = load_selected_hidx(split, suffix) # (sorted by construction)
	_ = selected_hidx.pop("criteria")
	if farther_sample_size>0:
		selected_fs = load_farther_sampled_ids(split, farther_sample_size, suffix) # ids valid in the nested for loops over all_img_paths then over selected_hidx
		consider = lambda data_id: data_id in selected_fs
		suffix = f'fs{farther_sample_size}_{suffix}'
	else:
		consider = lambda data_id: True
	print("Done loading the dataset!")

	# [validation (10k): 5s]
	# build look up table: pose_id -> (image_path, hidx, subject_identity)
	id_2_refs, data_id = {}, 0
	for img_path in tqdm(all_img_paths): # (sorted order)
		for h_ind, h_idx in enumerate(selected_hidx[img_path]): # (sorted order by construction)
			if consider(data_id): # limit construction to selected poses 
				id_2_refs[data_id] = (img_path, h_idx, annots[img_path][h_idx]["identity"])
			data_id += 1
	print(f"Done building the pose look-up table: parsed {data_id} poses, selected {len(id_2_refs)}.")
	

	# (2) GET MATRICES FOR SELECTION -------------------------------------------

	# [validation (10k): 12s]
	# get time difference between two poses
	# * B poses along rows, A along columns
	# * zero coefficients denote two poses that were not extracted from the same
	#   motion (ie. they were performed by 2 different persons, or in 2
	#   different videos)
	time_diff_mat = get_time_difference_mat(split, id_2_refs, suffix)

	# [validation (10k): 45s]
	# get posetext features for each pose
	retrieval_features = get_posesecript_features(split, farther_sample_size, suffix)

	# [validation (10k): quick]
	# get posecodes for each pose
	posecodes_intptt, max_posecodes = get_posecodes(split, id_2_refs, annots, suffix)


	# (3) SELECT ---------------------------------------------------------------

	# initializations
	ft_mat, ft_thresholds = compute_feature_matrix(retrieval_features)
	print(f"[{split}] No constraint: {ft_mat.shape[0]**2} pairs.")

	# sequence constraint
	s, time_diff_mat = sequence_constraint(kind, time_diff_mat)
	print(f"[{split}] Applying the sequence constraint: {s.sum().item()} pairs ({round(s.sum().item() /(s.shape[0]**2) * 100, 4)}%).")

	# feature constraint
	s = torch.logical_and(s, feature_constraint(ft_mat, ft_thresholds))
	print(f"[{split}] Applying the feature constraint: {s.sum().item()} pairs ({round(s.sum().item() /(s.shape[0]**2) * 100, 4)}%).")

	# posecode constraint
	s, pc_info, nb_zeroed_B_poses = posecode_constraint(kind, s, posecodes_intptt, ft_mat, max_posecodes)
	print(f"[{split}] Applying the posecode constraint: {s.sum().item()} pairs ({round(s.sum().item() /(s.shape[0]**2) * 100, 4)}%).")
	print(f"[{split}] Number of new poses B for which no pose A can be found to satisfy the posecode constraint: {nb_zeroed_B_poses}")
	print(f"[{split}] Number of pairs satisfying the constraints but that were chosen by default (# eligible A poses < PC_CONSTRAINT_WINDOW = {PC_CONSTRAINT_WINDOW}): {(pc_info < 0).sum().item()}")

	# pose unicity in role
	if kind == "in-sequence":
		s = pose_unicity_in_role_for_insequence_mining(s, time_diff_mat)
		print(f"[{split}] Applying the role unicity constraint: {s.sum().item()} pairs ({round(s.sum().item() /(s.shape[0]**2) * 100, 4)}%).")


	# (4) ORDER & SAVE ---------------------------------------------------------

	# annotation order
	pairs_to_annotate, s = get_ordered_annotation_pairs(kind, s, id_2_refs)
	# NOTE: at this point, `pairs_to_annotate` is:
	# 	* a torch tensor of shape (nb_pairs, 2)
	#	* formatted as: [pose B id, pose A id]
	# 	* with pose ids being local indices in the studied set!

	# format & save data
	metadata_select = pairs_to_annotate.t().unbind()
	pair_filepath = os.path.join(config.BEDLAM_PROCESSED_DIR, f"bedlam_fix_pairs_{split}_{kind}{'_'+suffix if suffix else ''}.pt")
	torch.save({
			"pairs": pairs_to_annotate,
			"local2global_pose_ids": list(id_2_refs.keys()),
			"ft_mat": ft_mat[metadata_select],
			"time_diff_mat": time_diff_mat[metadata_select] if kind == 'in-sequence' else None,
			"pc_info": pc_info[metadata_select],
			"max_posecodes": max_posecodes
		}, pair_filepath)

	print(f"[{split}] FINAL: {s.sum().item()} possible pairs (ie. {round(s.sum().item() /(s.shape[0]**2) * 100, 4)}% selected pairs)")
	print("Saved", pair_filepath)



################################################################################
################################################################################


if __name__ == "__main__":

	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument('--split', type=str, choices=('training', 'validation', 'test'), default='validation')
	parser.add_argument('--kind', type=str, choices=('in', 'out'), help='whether to mine in- or out-of- sequence pairs')
	parser.add_argument('--farther_sample', type=int, default=0, help='whether to mine within the set of farther sampled poses; size of the set')
	
	args = parser.parse_args()
	
	suffix = "try"
	select_pairs(args.split, f"{args.kind}-sequence", args.farther_sample, suffix)