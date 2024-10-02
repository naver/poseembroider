##############################################################
## poseembroider                                            ##
## Copyright (c) 2024                                       ##
## Institut de Robotica i Informatica Industrial, CSIC-UPC  ##
## and Naver Corporation                                    ##
## Licensed under the CC BY-NC-SA 4.0 license.              ##
## See project root for license details.                    ##
##############################################################

import os
from tqdm import tqdm
import torch
import numpy as np
import roma

import poseembroider.config as config
import poseembroider.utils as utils
import poseembroider.evaluator as evaluator
from poseembroider.datasets.bedlam_script import BEDLAMScript
from poseembroider.assisting_models.poseVAE.model_poseVAE import PoseVAE
from poseembroider.assisting_models.poseVAE.fid import FID
from poseembroider.assisting_models.poseVAE.loss import laplacian_nll, gaussian_nll

os.environ['TOKENIZERS_PARALLELISM'] = 'false'

OVERWRITE_RESULT = False


################################################################################

def load_model(model_path, device):
	
	assert os.path.isfile(model_path), "File {} not found.".format(model_path)
	
	# load checkpoint & model info
	ckpt = torch.load(model_path, 'cpu')
	
	# load model
	model = PoseVAE(latentD=ckpt['args'].latentD,
				 	num_body_joints=ckpt['args'].num_body_joints
					).to(device)
	model.load_state_dict(ckpt['model'])
	model.eval()
	print(f"Loaded model from (epoch {ckpt['epoch']}):", model_path)
	
	return model, None


def eval_model(model_path, dataset_version, fid_version, split='val'):
	
	device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

	# set seed for reproducibility (sampling for pose generation)
	torch.manual_seed(42)
	np.random.seed(42)

	# define result file
	fid_version, precision = get_evaluation_auxiliary_info(fid_version)
	result_filepath = evaluator.get_result_filepath(model_path, split, dataset_version, precision)

	# compute or load results
	if OVERWRITE_RESULT or not os.path.isfile(result_filepath):
		# load model
		model, _ = load_model(model_path, device)
		# load data
		if "bedlamscript" in dataset_version:
			fs_size = {"val":10000, "train":50000}[split]
			d = BEDLAMScript(version=dataset_version,
						split=split,
						reduced_set=fs_size,
						num_body_joints=model.pose_encoder.num_body_joints,
						item_format='p')
		else:
			raise NotImplementedError
		# evaluate
		results = compute_eval_metrics(model, d, fid_version, device)
		evaluator.save_results_to_file(results, result_filepath)
	else:
		results = evaluator.load_results_from_file(result_filepath)

	return {k:[v] for k, v in results.items()}


def get_evaluation_auxiliary_info(fid_version):
	precision = ""
	if fid_version:
		precision += f"_X{fid_version}X"
	return fid_version, precision


def compute_eval_metrics(model, dataset, fid_version, device):

	# initialize dataloader
	data_loader = torch.utils.data.DataLoader(
		dataset, sampler=None, shuffle=False,
		batch_size=32,
		num_workers=8,
		pin_memory=True,
		drop_last=False
	)

	# init body model
	body_model = utils.BodyModelSMPLX(config.SMPLX_BODY_MODEL_PATH)
	body_model.eval()
	body_model.to(device)
	
	# init FID
	if fid_version:
		fid = FID(version=fid_version, device=device)
		fid.extract_real_features(data_loader)
		fid.reset_gen_features()

	pose_metrics = {f'{k}_{v}': 0.0 for k in ['v2v', 'jts', 'rot'] for v in ['elbo', 'dist_avg'] \
					+ [f'dist_top{topk}' for topk in config.k_topk_reconstruction_values]}
	
	# compute metrics
	for batch in tqdm(data_loader):

		# data setup
		model_input = dict(
			poses = batch['pose'].to(device)
		)

		with torch.inference_mode():
			pose_metrics, _ = add_elbo_and_reconstruction(model_input, pose_metrics, model, body_model, output_distr_key="q_z", reference_pose_key="poses")
			if fid_version:
				fid.add_gen_features( model.sample_nposes(**model_input, n=1)['pose_body'] )

	# average over the dataset
	for k in pose_metrics: pose_metrics[k] /= len(dataset)

	# normalize the elbo (the same is done earlier for the reconstruction metrics)
	pose_metrics.update({'v2v_elbo':pose_metrics['v2v_elbo']/(body_model.J_regressor.shape[1] * 3),
						'jts_elbo':pose_metrics['jts_elbo']/(body_model.J_regressor.shape[0] * 3),
						'rot_elbo':pose_metrics['rot_elbo']/(model.pose_decoder.num_body_joints * 9)})

	# compute fid metric
	results = {'fid':-1}
	if fid_version:
		results.update({'fid': fid.compute()})
	results.update(pose_metrics)

	return results


def add_elbo_and_reconstruction(model_input, results, model, body_model, output_distr_key, reference_pose_key):
	"""
	Args:
		model_input: dict yielding the right arguments for the model forward
			functions.
		results: dict containing initial values for all elbo & reconstruction
	   		metrics.
		model: pose generative model
		output_distr_key: key to retrieve the output query (fusion) distribution
			from the output of the pose generative model
		reference_pose_key: key to the reference poses in the model_input dict

	Returns:
		results: updated with the measures made on the current batch
		output: result of the model forward function
	"""

	batch_size = model_input[reference_pose_key].shape[0]

	# generate pose
	output = model.forward(**model_input)

	# initialize bodies
	bm_rec = body_model(**utils.pose_data_as_dict(output['pose_body_pose']))
	bm_orig = body_model(**utils.pose_data_as_dict(model_input[reference_pose_key]))

	# a) compute elbos
	kld = torch.sum(torch.distributions.kl.kl_divergence(output['q_z'], output[output_distr_key]), dim=[1])
	results['v2v_elbo'] += (-laplacian_nll(bm_orig.vertices, bm_rec.vertices, model.decsigma_v2v).sum((1,2)) - kld).sum().detach().item() # (batch_size, nb_vertices, 3): first sum over the coeffs, substract the kld, then sum over the batch
	results['jts_elbo'] += (-laplacian_nll(bm_orig.joints, bm_rec.joints, model.decsigma_jts).sum((1,2)) - kld).sum().detach().item() # (batch_size, nb_joints, 3): first sum over the coeffs, substract the kld, then sum over the batch
	results['rot_elbo'] += (-gaussian_nll(output['pose_body_matrot_pose'].view(-1,3,3), roma.rotvec_to_rotmat(model_input[reference_pose_key].view(-1, 3)), model.decsigma_rot).view(batch_size, -1, 3, 3).sum((1,2,3)) - kld).sum().detach().item() # (batch_size, nb_joints, 3, 3): first sum over the coeffs, substract the kld, then sum over the batch

	# b) compute reconstructions
	# best pose out of nb_sample generated ones
	# -- sample several poses using the text
	generated_poses = model.sample_nposes(**model_input, n=config.nb_sample_reconstruction) # shape (batch_size, nb_sample, ...)
	bm_samples = body_model(**utils.pose_data_as_dict(generated_poses['pose_body'].flatten(0,1))) # flatten in (batch_size*nb_sample, sub_nb_joints*3)
	# -- compute reconstruction metrics for all samples
	v2v_dist = evaluator.L2multi(bm_samples.vertices.view(batch_size, config.nb_sample_reconstruction, -1, 3), bm_orig.vertices.unsqueeze(1)) # (batch_size, nb_sample)
	jts_dist = evaluator.L2multi(bm_samples.joints.view(batch_size, config.nb_sample_reconstruction, -1, 3), bm_orig.joints.unsqueeze(1)) # (batch_size, nb_sample)
	rot_dist = roma.rotmat_geodesic_distance(
					generated_poses['pose_body_matrot'].view(batch_size, config.nb_sample_reconstruction, -1, 3, 3),
					roma.rotvec_to_rotmat(model_input[reference_pose_key].view(-1,3)).view(batch_size, 1, -1, 3, 3)
				).mean(-1) * 180 / torch.pi # (batch_size, nb_sample), in degrees
	# -- extract reconstruction metrics:
	# * average --> mean along the sample axis; sum along the batch axis
	results['v2v_dist_avg'] += v2v_dist.mean(1).sum().detach().item() 
	results['jts_dist_avg'] += jts_dist.mean(1).sum().detach().item()
	results['rot_dist_avg'] += rot_dist.mean(1).sum().detach().item()
	# * top K --> get topk samples
	#   (dim=1 is the sample dimension; `largest' tells whether the
	#   higher the better; [0] allows to retrieve the actual values and
	#   not the indices); average values along the sample axis for the
	#   topk selected elements; then sum along the batch axis
	for topk in config.k_topk_reconstruction_values:
		results[f'v2v_dist_top{topk}'] += v2v_dist.topk(k=topk, dim=1, largest=False)[0].mean(1).sum().detach().item()
		results[f'jts_dist_top{topk}'] += jts_dist.topk(k=topk, dim=1, largest=False)[0].mean(1).sum().detach().item()
		results[f'rot_dist_top{topk}'] += rot_dist.topk(k=topk, dim=1, largest=False)[0].mean(1).sum().detach().item()

	return results, output


def display_results(results):
	metric_order = ['fid'] + [f'{x}_elbo' for x in ['jts', 'v2v', 'rot']] \
					+ [f'{k}_{v}' for k in ['jts', 'v2v', 'rot']
								for v in ['dist_avg'] + [f'dist_top{topk}' for topk in config.k_topk_reconstruction_values]]
	results = evaluator.scale_and_format_results(results)
	print(f"\n<model> & {' & '.join([results[m] for m in metric_order])} \\\\\n")


################################################################################

if __name__=="__main__":

	# added special arguments
	evaluator.eval_parser.add_argument('--fid', type=str, default='', help='Version of the fid to use for evaluation.')

	args = evaluator.eval_parser.parse_args()
	args = evaluator.get_full_model_path(args)

	# compute results
	ret = eval_model(args.model_path, dataset_version=args.dataset, fid_version=args.fid, split=args.split)

	# display results
	print(ret)
	display_results(ret)