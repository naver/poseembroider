import argparse
import os
from tqdm import tqdm
import re
import json
import torch
import evaluate
import random

import poseembroider.config as config
import poseembroider.utils as utils

os.environ['TOKENIZERS_PARALLELISM'] = 'false'


################################################################################
## PARSER
################################################################################

eval_parser = argparse.ArgumentParser(description='Evaluation parameters.')
eval_parser.add_argument('--model_path', type=str, help='Path to the model.')
eval_parser.add_argument('--checkpoint', default='best', choices=('best', 'last'), help='Checkpoint to choose if model path is incomplete.')
eval_parser.add_argument('--dataset', type=str,  help='Evaluation dataset.')
eval_parser.add_argument('--split', default="val", type=str, help='Split to evaluate.')
eval_parser.add_argument('--special_suffix', default="", type=str, help='Special suffix for the result filename.')


################################################################################
## UTILS
################################################################################


## Normalize result values
################################################################################

NLP_metric_coeff = 100
DIST_coeff = 1000 # get results in milimeters

def scale_and_format_results(ret):
	"""
	Args:
		dict{k:list of numbers}
	Returns:
		dict{k:string}
	"""
	for k in ret:
		# convert each sub-element to string
		if k=='fid':
			ret[k] = ['%.2f' % x for x in ret[k]]
		elif re.match(r'.*(jts|v2v)_dist.*', k) or re.match(r'.*JtposDist.*', k) or ('MPJE' in k):
			ret[k] = ['%.0f' % (x*DIST_coeff) for x in ret[k]]
		elif k in ['bleu', 'rougeL', 'meteor']:
			ret[k] = ['%.1f' % (x*NLP_metric_coeff) for x in ret[k]]
		elif 'R@' in k or 'mRecall' in k:
			ret[k] = ['%.1f' % x for x in ret[k]]
		else:
			ret[k] = ['%.1f' % x for x in ret[k]]
		# case: average over runs
		ret[k] = ret[k][0] if len(ret[k])==1 else "%s \\tiny{${\pm}$ %s}" % tuple(ret[k])
	return ret


def mean_list(data):
	return sum(data)/len(data)


## Get info from model
################################################################################

def get_epoch(model_path=None, ckpt=None):
	assert model_path or ckpt, "Must provide at least one argument!"
	if ckpt:
		return ckpt['epoch']
	else:
		return torch.load(model_path, 'cpu')['epoch']


def get_full_model_path(args, control_measures=None):
	if ".pth" not in args.model_path and (not control_measures or args.model_path not in control_measures):
		args.model_path = os.path.join(args.model_path, f"checkpoint_{args.checkpoint}.pth")
		print(f"Checkpoint not specified. Using {args.checkpoint} checkpoint.")
	return args


## Compute metrics
################################################################################

def L2multi(x, y):
	# x: torch tensor of size (*,N,P,3) or (*,P,3)
	# y: torch tensors of size (*,P,3)
	# return: torch tensor of size (*,N,1) or (*,1)
	return torch.linalg.norm(x-y, dim=-1).mean(-1)


def x2y_recall_metrics(x_features, y_features, k_values, sstr=""):
	"""
	Args:
		x_features, y_features: shape (batch_size, latentD)
	"""

	# initialize metrics
	nb_x = len(x_features)
	sstrR = sstr + 'R@%d'
	recalls = {sstrR%k:0 for k in k_values}

	# evaluate for each query x
	for x_ind in tqdm(range(nb_x)):
		# compute scores
		scores = x_features[x_ind].view(1, -1).mm(y_features.t())[0].cpu()
		# sort in decreasing order
		_, indices_rank = scores.sort(descending=True)
		# update recall metrics
		# (the rank of the ground truth target is given by the position of x_ind
		# in indices_rank, since ground truth x/y associations are identified
		# through indexing)
		GT_rank = torch.where(indices_rank == x_ind)[0][0].item()
		for k in k_values:
			recalls[sstrR%k] += GT_rank < k

	# average metrics
	recalls = {sstrR%k: recalls[sstrR%k]/nb_x*100.0 for k in k_values}
	return recalls


def textret_metrics(all_text_embs, all_pose_embs):

	n_queries = all_text_embs.shape[0]

	all_gt_rank = torch.zeros(n_queries)
	for i in tqdm(range(n_queries)):
		# average the process over a number of repetitions
		for _ in range(config.r_precision_n_repetitions):
			# randomly select config.sample_size_r_precision elements
			# (including the query)
			selected = random.sample(range(n_queries), config.sample_size_r_precision)
			selected = [i] + [s for s in selected if s != i][:config.sample_size_r_precision - 1]
			# compute scores (use the same as for model training: similarity instead of the Euclidean distance)
			scores = all_text_embs[i].view(1,-1).mm(all_pose_embs[selected].t())[0].cpu()
			# rank
			_, indices_rank = scores.sort(descending=True)
			# compute recalls (GT is always expected in position 0)
			GT_rank = torch.where(indices_rank == 0)[0][0].item()
			all_gt_rank[i] += GT_rank
	all_gt_rank /= config.r_precision_n_repetitions

	ret = {f'ret_r{k}_prec': (all_gt_rank < k).sum().item()/n_queries*100 for k in config.k_topk_r_precision}
	
	return ret


def compute_eval_metrics_p2t_t2p(model, dataset, device, infer_features_func, compute_loss=False, loss_func=None):
	
	# get data features
	poses_features, texts_features = infer_features_func(model, dataset, device)
	
	# poses-2-text matching
	p2t_recalls = x2y_recall_metrics(poses_features, texts_features, config.k_recall_values, sstr="p2t_")
	# text-2-poses matching
	t2p_recalls = x2y_recall_metrics(texts_features, poses_features, config.k_recall_values, sstr="t2p_")
	# r-precision
	rprecisions = textret_metrics(texts_features, poses_features)

	# gather metrics
	recalls = {"mRecall": (sum(p2t_recalls.values()) + sum(t2p_recalls.values())) / (2 * len(config.k_recall_values))}
	recalls.update(p2t_recalls)
	recalls.update(t2p_recalls)
	recalls.update(rprecisions)

	# loss
	if compute_loss:
		score_t2p = texts_features.mm(poses_features.t())
		loss = loss_func(score_t2p*model.loss_weight)
		loss_value = loss.item()
		return recalls, loss_value

	return recalls


def compute_NLP_metrics(ground_truth_texts, generated_texts):

	results = {}
	all_keys = list(ground_truth_texts.keys())

	for mname, mtag in zip(["bleu", "rouge", "meteor", "bertscore"], ["bleu", "rougeL", "meteor", "precision"]):
		metric = evaluate.load(mname)
		metric.add_batch(references=[ground_truth_texts[k] for k in all_keys], predictions=[generated_texts[k][0] for k in all_keys])
		if mname == "bertscore":
			tmp = metric.compute(model_type="distilbert-base-uncased") # not yet aggregated over the whole dataset!
			results[mname] = mean_list(tmp[mtag])
			print(f"BertScore: hashcode {tmp['hashcode']}")
		else:
			results[mtag] = metric.compute()[mtag] # aggregated over the whole dataset

	return results


def procrustes_align(S1: torch.Tensor, S2: torch.Tensor, return_transformation=False):
	"""
	from https://github.com/shubham-goel/4D-Humans/blob/124e8b2dc041fdd7e02f879ce851952cb810636b/hmr2/utils/pose_utils.py#L9
	Computes a similarity transform (sR, t) in a batched way that takes
	a set of 3D points S1 (B, N, 3) closest to a set of 3D points S2 (B, N, 3),
	where R is a 3x3 rotation matrix, t 3x1 translation, s scale.
	i.e. solves the orthogonal Procrutes problem.
	Args:
		S1 (torch.Tensor): First set of points of shape (B, N, 3).
		S2 (torch.Tensor): Second set of points of shape (B, N, 3).
	Returns:
		(torch.Tensor): The first set of points after applying the similarity transformation.
	"""

	batch_size = S1.shape[0]
	S1 = S1.permute(0, 2, 1)
	S2 = S2.permute(0, 2, 1)
	# 1. Remove mean.
	mu1 = S1.mean(dim=2, keepdim=True)
	mu2 = S2.mean(dim=2, keepdim=True)
	X1 = S1 - mu1
	X2 = S2 - mu2

	# 2. Compute variance of X1 used for scale.
	var1 = (X1**2).sum(dim=(1,2))

	# 3. The outer product of X1 and X2.
	K = torch.matmul(X1, X2.permute(0, 2, 1))

	# 4. Solution that Maximizes trace(R'K) is R=U*V', where U, V are singular vectors of K.
	U, s, V = torch.svd(K)
	Vh = V.permute(0, 2, 1)

	# Construct Z that fixes the orientation of R to get det(R)=1.
	Z = torch.eye(U.shape[1], device=U.device).unsqueeze(0).repeat(batch_size, 1, 1)
	Z[:, -1, -1] *= torch.sign(torch.linalg.det(torch.matmul(U, Vh)))

	# Construct R.
	R = torch.matmul(torch.matmul(V, Z), U.permute(0, 2, 1))

	# 5. Recover scale.
	trace = torch.matmul(R, K).diagonal(offset=0, dim1=-1, dim2=-2).sum(dim=-1)
	scale = (trace / var1).unsqueeze(dim=-1).unsqueeze(dim=-1)

	# 6. Recover translation.
	t = mu2 - scale*torch.matmul(R, mu1)

	if return_transformation:
		return (R, t, scale)
	else:
		# 7. Error:
		S1_hat = scale*torch.matmul(R, S1) + t
		return S1_hat.permute(0, 2, 1)


## Save results
################################################################################

def get_result_filepath(model_path, split, dataset_version, precision="",
							controled_task="", special_end_suffix=""):
	if ".pth" in model_path:
		get_res_file = os.path.join(os.path.dirname(model_path),
						f"result_{split}_{dataset_version}{precision}" + \
						f"_{get_epoch(model_path=model_path)}{special_end_suffix}.txt")
	else: # control measure
		# NOTE: the `model_path` is actually the name of the controled measure
		get_res_file = os.path.join(config.GENERAL_EXP_OUTPUT_DIR,
						f"result_{controled_task}_control_measures_{model_path}" + \
						f"_{split}_{dataset_version}{precision}" + \
						f"{special_end_suffix}.txt")
	return get_res_file


def save_results_to_file(data, filename_res):
	utils.write_json(data, filename_res)
	print("Saved file:", filename_res)


def load_results_from_file(filename_res):
	with open(filename_res, "r") as f:
		data = json.load(f)
		data = {k:float(v) if type(v) is not dict
							else {kk:float(vv) for kk,vv in v.items()}
					for k, v in data.items()} # parse values
		print("Load results from", filename_res)
	return data