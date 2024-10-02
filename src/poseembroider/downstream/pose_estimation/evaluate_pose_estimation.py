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
import roma

from text2pose.encoders.tokenizers import get_tokenizer_name

import poseembroider.config as config
import poseembroider.utils as utils
import poseembroider.evaluator as evaluator
from poseembroider.datasets.bedlam_script import BEDLAMScript
from poseembroider.datasets.threedpw import ThreeDPW
from poseembroider.downstream.pose_estimation.model_pose_estimation import HPSEstimator
from poseembroider.augmentations import DataProcessingModule

os.environ['TOKENIZERS_PARALLELISM'] = 'false'

OVERWRITE_RESULT = False

################################################################################

def load_model(model_path, device='cpu'):

	assert os.path.isfile(model_path), "File {} not found.".format(model_path)
	
	# load checkpoint & model info
	ckpt = torch.load(model_path, 'cpu')
	
	# extract some properties
	path_to_pretrained_representation_model = utils.read_json(config.PRETRAINED_MODEL_DICT)[ckpt['args'].pretrained_representation_model]

	# load model
	model = HPSEstimator(
			num_body_joints=ckpt['args'].num_body_joints,
			num_shape_coeffs=ckpt['args'].num_shape_coeffs,
			predict_bodyshape = ckpt['args'].predict_bodyshape,
			# -- about the representation model
			encoder_latentD=ckpt['args'].latentD,
			path_to_pretrained_representation_model=path_to_pretrained_representation_model
			)
	
	model.to(device)

	# careful load of the state dict	
	missing_keys, unexpected_keys = model.load_state_dict(ckpt['model'], strict=False)
	# Ignore missing keys related to the 'representation_model', assuming the
	# reason for this is that the previous training made use of the cached
	# features. Ensure these are the only keys missing.
	missing_keys = [k for k in missing_keys if not k.startswith("representation_wrapper.representation_model.")]
	assert len(missing_keys) == 0 and len(unexpected_keys) == 0, "Key mismatch when loading state_dict."
	
	model.eval()
	print(f"Loaded model from (epoch {ckpt['epoch']}):", model_path)

	# get config info
	text_encoder_name = model.representation_wrapper.representation_model.text_encoder_name
	tokenizer_name = "distilbertUncased" if text_encoder_name=="posetext" else get_tokenizer_name(text_encoder_name)
	
	return model, tokenizer_name


def eval_model(model_path, dataset_version, split='val', human_visibility='any', text_ind=0):
	
	device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

	precision = ""
	# add information in the result filepath if the results were obtained on
	# a specific set
	if human_visibility!="any":
		if ("bedlamscript" in dataset_version) and human_visibility!="any":
			precision += f"_{human_visibility}HumanVisibilitySet"
		else:
			print("This dataset does not allow sample filtering based on the proportion of human visibility in images.")

	# define result file
	result_filepath = evaluator.get_result_filepath(model_path, split, dataset_version, precision)

	# compute or load results
	if OVERWRITE_RESULT or not os.path.isfile(result_filepath):
		# load model
		model, tokenizer_name = load_model(model_path, device)
		img_processing_scheme = utils.get_img_processing_scheme(model.representation_wrapper.representation_model.image_encoder_name)
		# load data
		if "bedlamscript" in dataset_version:
			fs_size = {"val":10000, "train":50000}[split]
			d = BEDLAMScript(version=dataset_version,
						split=split,
						reduced_set=fs_size,
						num_body_joints=model.num_body_joints,
						num_shape_coeffs=model.num_shape_coeffs,
						tokenizer_name=tokenizer_name,
						text_index=text_ind,
						img_processing_scheme=img_processing_scheme,
						human_visibility=human_visibility)
		elif "threedpw" in dataset_version:
			d = ThreeDPW(version=dataset_version,
						split=split,
						num_body_joints=model.num_body_joints,
						num_shape_coeffs=model.num_shape_coeffs,
						img_processing_scheme=img_processing_scheme,
						item_format='ip')
		else:
			raise NotImplementedError
		# evaluate
		results = compute_eval_metrics_for_all_query_types(model, d, device)
		evaluator.save_results_to_file(results, result_filepath)
	else:
		results = evaluator.load_results_from_file(result_filepath)

	return {k:[v] for k, v in results.items()}


def compute_eval_metrics_for_all_query_types(model, dataset, device, pose2pose=False):

	# init body model
	body_model = utils.BodyModelSMPLX(config.SMPLX_BODY_MODEL_PATH, num_betas=model.num_shape_coeffs)
	body_model.eval()
	body_model.to(device)

	# list the different sets of input types we can evaluate on, depending on
	# the available modalities in the dataset 
	all_input_type = []
	if 'i' in dataset.item_format:
		all_input_type.append(["images"])
	if 't' in dataset.item_format:
		all_input_type.append(["texts_tokens", "texts_lengths"])
	if 'i' in dataset.item_format and 't' in dataset.item_format:
		all_input_type.append(["images", "texts_tokens", "texts_lengths"])
	if pose2pose:
		# to check how well the core representation makes it possible to
		# "reconstruct" the pose
		all_input_type.append(["poses"])

	# evaluate the model on the different input sets
	results = {}
	input_type_to_str = lambda input_type: "-".join(sorted(set([it[0] for it in input_type]))) # code for the input subset
	for input_type in all_input_type:
		r = compute_eval_metrics(model, dataset, device, body_model, representation_model_input_types=input_type)
		prefix = input_type_to_str(input_type)
		results.update({f'{prefix}_{k}':v for k,v in r.items()})
		display_results({k:[v] for k, v in r.items()}, row_name=prefix)
	
	return results


def compute_eval_metrics(model, dataset, device, body_model, 
						 representation_model_input_types=['images']):

	# decoding setting
	# NOTE: if the text carries information about the body shape, you can
	# also predict the body shape from text input (please use the same setting
	# as during training)
	can_predict_shape = 'images' in representation_model_input_types

	# initialize dataloader
	batch_size = 32
	data_loader = torch.utils.data.DataLoader(
		dataset, sampler=None, shuffle=False,
		batch_size=batch_size,
		num_workers=8,
		pin_memory=True,
		drop_last=False
	)
	data_processing = DataProcessingModule(phase="eval", nb_joints=None, img_processing_scheme=dataset.img_processing_scheme) # we only care about image processing

	# initialize results
	ret = {k:torch.zeros(len(dataset)).to(device) for k in ["posePredRotDist", "posePredJtposDist", "posePredVertposDist", "paMPJE"]}
	joint_set = torch.concat([torch.arange(22), torch.arange(25,25+30)])[:dataset.num_body_joints] # only main body joints (including global rotation) & hand joints

	# predict pose, batch by batch
	for i, item in tqdm(enumerate(data_loader)):

		# load data
		item = {k:v.to(device) if k not in ["indices", "dataset"] else v for k,v in item.items() } # set on device
		item = data_processing(item) # image processing
		if 'texts_tokens' in item:
			item["texts_tokens"] = item["texts_tokens"][:,:max(item["texts_lengths"])] # truncate within the batch, based on the longest text 
		n_samples = len(item["indices"])
		s = slice(i*batch_size, i*batch_size+n_samples)

		# predict pose & shape
		with torch.no_grad() and torch.inference_mode():
			pose_pred_rotmat, pred_betas = model(item=item, representation_model_input_types=representation_model_input_types) # `pose_pred_rotmat`: shape (n_samples, n_joints, 3, 3) ; `pred_betas`: (n_samples, n_betas) or None
		pose_pred_rotvec = roma.rotmat_to_rotvec(pose_pred_rotmat) # shape (n_samples, n_joints, 3)

		# prepare ground truth
		pose_data_rotvec = item["poses"] # axis-angle representation; shape (n_samples, n_joints, 3)
		pose_data_rotmat = roma.rotvec_to_rotmat(pose_data_rotvec) # shape (n_samples, n_joints, 3, 3)

		if model.predict_bodyshape and can_predict_shape:
			betas_data = item["shapes"]
			betas_pred = pred_betas
		else:
			betas_data, betas_pred = None, None

		# compute distances
		# -- joint rotation
		ret["posePredRotDist"][s] = roma.rotmat_geodesic_distance(pose_pred_rotmat, pose_data_rotmat).mean(-1) * 180.0 / torch.pi # mean over n_joints, in degrees, shape (n_samples)
		# -- joint position
		bm_orig = body_model(**utils.pose_data_as_dict(pose_data_rotvec, code_base="smplx"), betas=betas_data)
		bm_pred = body_model(**utils.pose_data_as_dict(pose_pred_rotvec, code_base="smplx"), betas=betas_pred)
		ret["posePredJtposDist"][s] = evaluator.L2multi(bm_pred.joints[:,joint_set], bm_orig.joints[:,joint_set]) # shape (nb_sample)
		ret["posePredVertposDist"][s] = evaluator.L2multi(bm_pred.vertices, bm_orig.vertices) # shape (nb_sample)
		pa_joints_pred = evaluator.procrustes_align(bm_pred.joints[:,joint_set], bm_orig.joints[:,joint_set])
		ret["paMPJE"][s] = evaluator.L2multi(pa_joints_pred, bm_orig.joints[:,joint_set]) # shape (nb_sample)

	# average over the dataset
	ret = {k:v.mean().item() for k,v in ret.items()}

	return ret


def display_results(results, row_name="<model>"):

	metric_order = ["paMPJE", "posePredRotDist", "posePredJtposDist", "posePredVertposDist"]
	print("Metric order: " + ' & '.join(metric_order))

	# get all prefixes
	prefixes = [k.replace("paMPJE", "") for k in results if "paMPJE" in k]
	# group data per prefix and display per group
	for p in prefixes:
		sub_results = {k.replace(p, ""):v for k,v in results.items() if p in k}
		sub_results = evaluator.scale_and_format_results(sub_results)
		print(f"\n{row_name} ({p[:-1]}) & {' & '.join([sub_results[m] for m in metric_order])} \\\\\n") # p[:-1] to remove the trailing "_" caracter


################################################################################

if __name__ == '__main__':

	# added special arguments
	evaluator.eval_parser.add_argument('--human_visibility', choices=('any', 'full', 'partial'), default='any', help='subset of images to consider (for datasets accepting the `human_visibility` argument).')
	
	args = evaluator.eval_parser.parse_args()
	args = evaluator.get_full_model_path(args)

	# compute results
	ret = eval_model(args.model_path, dataset_version=args.dataset, split=args.split, human_visibility=args.human_visibility)

	# display results
	print(ret)
	display_results(ret)