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

from text2pose.encoders.tokenizers import Tokenizer, get_tokenizer_name

import poseembroider.config as config
import poseembroider.utils as utils
import poseembroider.evaluator as evaluator
from poseembroider.datasets.bedlam_fix import BEDLAMFix
from poseembroider.datasets.posefix import PoseFix
from poseembroider.augmentations import DataProcessingModule
from poseembroider.downstream.instruction_generation.model_instruction_generation import InstructionGenerator
from poseembroider.assisting_models.pairText.utils_pairText import load_model as load_textret_model

os.environ['TOKENIZERS_PARALLELISM'] = 'false'

CONTROL_MEASURES = ["GT", "random"]
OVERWRITE_RESULT = False


################################################################################

def load_model(model_path, device):

	assert os.path.isfile(model_path), "File {} not found.".format(model_path)
	
	# load checkpoint & model info
	ckpt = torch.load(model_path, 'cpu')

	# extract some properties
	text_decoder_name = ckpt['args'].text_decoder_name
	path_to_pretrained_representation_model = utils.read_json(config.PRETRAINED_MODEL_DICT)[ckpt['args'].pretrained_representation_model]

	# load model
	model = InstructionGenerator(
								num_body_joints=ckpt['args'].num_body_joints,
								comparison_latentD=ckpt['args'].comparison_latentD,
								comparison_module_mode=ckpt['args'].comparison_module_mode,
								text_decoder_name=text_decoder_name,
								transformer_mode=ckpt['args'].transformer_mode,
								decoder_latentD=ckpt['args'].decoder_latentD,
								decoder_nhead=ckpt['args'].decoder_nhead,
								decoder_nlayers=ckpt['args'].decoder_nlayers,
								# -- about the representation model
								encoder_latentD=ckpt['args'].latentD,
								path_to_pretrained_representation_model=path_to_pretrained_representation_model,
								).to(device)

	# careful load of the state dict	
	missing_keys, unexpected_keys = model.load_state_dict(ckpt['model'], strict=False)
	# Ignore missing keys related to the 'representation_model', assuming the reason
	# for this is that the previous training made use of the cached
	# features. Ensure these are the only keys missing.
	missing_keys = [k for k in missing_keys if not k.startswith("representation_wrapper.representation_model.")]
	assert len(missing_keys) == 0 and len(unexpected_keys) == 0, "Key mismatch when loading state_dict."
	
	model.eval()
	print(f"Loaded model from (epoch {ckpt['epoch']}):", model_path)

	return model, get_tokenizer_name(text_decoder_name)


def eval_model(model_path, dataset_version, textret_model_version=None, split='val', human_visibility='any', pair_kind='any'):
	""""
	model_path: true path to the text generation model to evaluate, or one of
				CONTROL_MEASURES to compute control measures.
	"""
	
	device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

	# determine which metric to compute & get paths to required evaluation models
	control_measures = model_path if model_path in CONTROL_MEASURES else False
	textret_model_path, precision = get_evaluation_model_paths(textret_model_version)

	# add information in the result filepath if the results were obtained on
	# a specific set
	if human_visibility!="any":
		if "bedlamfix" in dataset_version:
			precision += f"_{human_visibility}HumanVisibilitySet"
		else:
			print("This dataset does not allow sample filtering based on the proportion of human visibility in images.")
	if pair_kind!="any":
		precision += f"_{pair_kind}Seq"

	# define result file
	result_filepath = evaluator.get_result_filepath(model_path, split, dataset_version, precision, controled_task="modifier_generation", special_end_suffix=args.special_suffix)
	
	# compute or load results
	if OVERWRITE_RESULT or not os.path.isfile(result_filepath):
		# load models
		if control_measures:
			model, tokenizer_name, img_processing_scheme = None, None, None
		else:
			model, tokenizer_name = load_model(model_path, device)
			img_processing_scheme = utils.get_img_processing_scheme(model.representation_wrapper.representation_model.image_encoder_name)
		textret_model, tokenizer_name_textret_model = get_evaluation_models(device, textret_model_path)
		# load data
		if "bedlamfix" in dataset_version:
			fs_size = {"val":10000, "train":50000}[split]
			d = BEDLAMFix(version=dataset_version,
				 		split=split,
						reduced_set=fs_size,
						num_body_joints=model.representation_wrapper.num_body_joints,
						tokenizer_name=tokenizer_name,
						text_index=0,
						img_processing_scheme=img_processing_scheme,
						pair_kind=pair_kind,
						human_visibility=human_visibility)
		elif "posefix" in dataset_version:
			d = PoseFix(version=dataset_version,
			   			split=split,
						num_body_joints=model.representation_wrapper.num_body_joints,
						tokenizer_name=tokenizer_name,
						text_index=0,
						item_format='pt',
						pair_kind=pair_kind)
		else:
			raise NotImplementedError
		# evaluate
		if control_measures:
			results = compute_eval_metrics(model, d, device,
					textret_model=textret_model,
					tokenizer_name_textret_model=tokenizer_name_textret_model,
					control_measures=control_measures)
		else:
			results = compute_eval_metrics_for_all_query_types(model, d, device,
					textret_model=textret_model,
					tokenizer_name_textret_model=tokenizer_name_textret_model)
		evaluator.save_results_to_file(results, result_filepath)
	else:
		results = evaluator.load_results_from_file(result_filepath)

	return {k:[v] for k, v in results.items()}


def get_evaluation_model_paths(textret_model_version=None):
	precision = ""
	textret_model_path = None
	if textret_model_version is not None:
		textret_model_path = utils.read_json(config.PRETRAINED_MODEL_DICT)[textret_model_version]
		assert os.path.isfile(textret_model_path), "Text-to-pose-pair retrieval model checkpoint not found: " + textret_model_path
		print("Using the following text-to-pose-pair retrieval model for evaluation:", textret_model_path)
		precision += f"_Z{textret_model_version}Z"
	return textret_model_path, precision


def get_evaluation_models(device, textret_model_path=None):
	textret_model, tokenizer_name_textret_model = None, None # default
	# load models for metrics
	if textret_model_path is not None:
		textret_model, tokenizer_name_textret_model = load_textret_model(textret_model_path, device)
	return textret_model, tokenizer_name_textret_model


def compute_eval_metrics_for_all_query_types(model, dataset, device,
											 textret_model=None,
											 tokenizer_name_textret_model=None):

	# list the different sets of input types we can evaluate on, depending on
	# the available modalities in the dataset 
	all_input_type = []
	if 'i' in dataset.item_format:
		all_input_type.append(["images_A", "images_B"]) # image image
	if 'p' in dataset.item_format:
		all_input_type.append(["poses_A", "poses_B"]) # pose pose
	if 'ip' in dataset.item_format:
		all_input_type += [["poses_A", "images_B"], # pose image
					 	   ["images_A", "poses_B"]] # image pose
	if 'ipt' in dataset.item_format:
		# it's also possible to consider the following case:
		all_input_type += [["poses_A", "images_A", "poses_B", "images_B"]] # all possible modalities together

	# evaluate the model on the different input sets
	input_type_to_str = lambda input_type: "-".join([f'{it.split("_")[0][0]}{it.split("_")[1]}' for it in input_type])
	results = {}
	for input_type in all_input_type:
		r = compute_eval_metrics(model, dataset, device,
				representation_model_input_types=input_type,
				textret_model=textret_model,
				tokenizer_name_textret_model=tokenizer_name_textret_model)
		prefix = input_type_to_str(input_type)
		results.update({f'{prefix}_{k}':v for k,v in r.items()})
		display_results({k:[v] for k, v in r.items()}, row_name=prefix)
	
	return results


def compute_eval_metrics(model, dataset, device,
						 representation_model_input_types=["poses_A", "poses_B"],
						 textret_model=None, tokenizer_name_textret_model=None,
						 control_measures=False):
	"""
	control_measures: (False|one of CONTROL_MEASURES)
	"""
	
	# initialize dataloader
	batch_size = 32
	data_loader = torch.utils.data.DataLoader(
		dataset, sampler=None, shuffle=False,
		batch_size=batch_size,
		num_workers=8,
		pin_memory=True,
		drop_last=False
	)
	if not control_measures:
		dataset.load_raw_texts() # load raw ground truth text (not just their tokenized version)
	data_processing = DataProcessingModule(phase="eval", nb_joints=None, img_processing_scheme=dataset.img_processing_scheme) # we only care about image processing

	# check that we can indeed use the textret model
	# (that model needs 3D poses input)
	if (tokenizer_name_textret_model is not None) and ('p' not in dataset.item_format):
		print(f"The dataset does not yield 3D poses (current item_format: {dataset.item_format})." + \
				"Thus, the pair-to-text retrieval model cannot be used for evaluation.")
		# deactivate the use of the textret model 
		textret_model=None
		tokenizer_name_textret_model=None

	# initialize results
	ground_truth_texts = {}
	generated_texts = {}
	retrieval_metrics = {f'ret_r{k}_prec': 0.0 for k in config.k_topk_r_precision}
	results = {'avg_likelihood': 0.0}

	# prepare grounds for retrieval metrics
	if textret_model is not None:
		n_queries = len(dataset)
		tokenizer_textret_model = Tokenizer(tokenizer_name_textret_model)
		all_pose_embs = torch.zeros(n_queries, textret_model.latentD).to(device)
		all_text_embs = torch.zeros(n_queries, textret_model.latentD).to(device)

	# generate text, batch by batch
	for i_batch, item in tqdm(enumerate(data_loader)):
		
		# set up data
		ground_truth_texts.update({index.item(): dataset.get_all_raw_texts(index.item()) for index in item['indices']})
		item = data_processing(item) # process images
		input_dict = {k:v.to(device) if k not in ["indices", "dataset"] else v for k,v in item.items() } # set on device
		this_batch_size = len(item['indices']) # may be different from batch_size, due to incomplete batches

		with torch.no_grad() and torch.inference_mode():

			# get text to evaluate
			if control_measures:
				if control_measures == "GT":
					decoded_texts = [ground_truth_texts[index.item()][0] for index in item['indices']] # select the first reference text
				elif control_measures == "random":
					# shuffle ground truth by 1
					decoded_texts = [ground_truth_texts[index.item()][0] for index in item['indices']] # select the first reference text
					decoded_texts = decoded_texts[1:] + decoded_texts[:1]
			else:
				# generate text
				decoded_texts, likelihood_scores = model.generate_text(item=input_dict, representation_model_input_types=representation_model_input_types)
				results["avg_likelihood"] += likelihood_scores.sum().item()
			generated_texts.update({index.item():[decoded_texts[i]] for i, index in enumerate(item['indices'])})
	
			# compute and store features for retrieval metrics
			if textret_model:

				# tokenize & padd decoded texts
				caption_tokens_, caption_lengths_ = tokenizer_textret_model.assemble_raw_texts(decoded_texts)
				caption_tokens_ = caption_tokens_.to(device)
				caption_lengths_ = caption_lengths_.to(device)
				# compute embeddings
				pose_embs, text_embs = textret_model(input_dict["poses_A"], caption_tokens_, caption_lengths_, input_dict["poses_B"])
				all_pose_embs[i_batch*batch_size:i_batch*batch_size+this_batch_size] = pose_embs
				all_text_embs[i_batch*batch_size:i_batch*batch_size+this_batch_size] = text_embs

	# average over the dataset
	for k in ['avg_likelihood']: results[k] /= len(dataset)

	# compute retrieval metrics
	if textret_model is not None:
		retrieval_metrics = evaluator.textret_metrics(all_text_embs, all_pose_embs)

	# compute NLP metrics
	nlp_metrics = evaluator.compute_NLP_metrics(ground_truth_texts, generated_texts)

	# gather results
	results.update(retrieval_metrics)
	results.update(nlp_metrics)

	return results


def display_results(results, row_name="<model>", multi_results=False):
	metric_order = [f'ret_r{k}_prec' for k in config.k_topk_r_precision] \
					+ ['bleu', 'rougeL', 'meteor']
	
	if multi_results:
		# get all prefixes
		prefixes = [k.replace("bleu", "") for k in results if "bleu" in k]
		# group data per prefix and display per group
		for p in prefixes:
			sub_results = {k.replace(p, ""):v for k,v in results.items() if p in k}
			sub_results = evaluator.scale_and_format_results(sub_results)
			print(f"\n{row_name} ({p[:-1]}) & {' & '.join([sub_results[m] for m in metric_order])} \\\\\n")
	else:
		results = evaluator.scale_and_format_results(results)
		print(f"\n{row_name} & {' & '.join([results[m] for m in metric_order])} \\\\\n")


################################################################################

if __name__ == '__main__':

	# added special arguments
	evaluator.eval_parser.add_argument('--textret_model', type=str, help="Shortname of the text-to-pose-pair retrieval model to use for computing top R-precision metrics.")
	evaluator.eval_parser.add_argument('--human_visibility', choices=('any', 'full', 'partial'), default='any', help='subset of images to consider (for datasets accepting the `human_visibility` argument).')
	evaluator.eval_parser.add_argument('--pair_kind', default='any', choices=('in', 'out', 'any'), help='kind of pairs to consider (in-sequence, out-of-sequence or both)') # NOTE: currently only valid for datasets deriving from the TriModalDatasetFix class
	
	args = evaluator.eval_parser.parse_args()
	args = evaluator.get_full_model_path(args, CONTROL_MEASURES)
	
	# compute results
	ret = eval_model(args.model_path,
					dataset_version=args.dataset,
					split=args.split,
					textret_model_version=args.textret_model,
					human_visibility=args.human_visibility,
					pair_kind=args.pair_kind
	)

	# display results
	print(ret)
	display_results(ret, multi_results=True) # treat all sub results at once