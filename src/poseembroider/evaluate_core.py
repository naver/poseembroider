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
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

from text2pose.encoders.tokenizers import get_tokenizer_name

import poseembroider.config as config
import poseembroider.utils as utils
import poseembroider.evaluator as evaluator
from poseembroider.model import PoseEmbroider, Aligner
from poseembroider.augmentations import DataProcessingModule
from poseembroider.datasets.bedlam_script import BEDLAMScript
from poseembroider.datasets.posescript import PoseScript
from poseembroider.datasets.threedpw import ThreeDPW

OVERWRITE_RESULT = False


################################################################################

def carefully_load_state_dict(model, state_dict):
	missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
	assert len(unexpected_keys)==0, 'Unexpected keys in state_dict: "'+'", "'.join(unexpected_keys) + '"'
	# filter out keys from pretrained frozen encoders
	if len(missing_keys):
		for x in ["image", "pose", "text"]:
			missing_keys = [k for k in missing_keys if not k.startswith(f"{x}_encoder.pretrained_{x}_encoder.")]
		print("NOTE: weights of the pretrained modality encoders were not found in the provided state_dict, however they may have been initialized beforehand.")
	assert len(missing_keys)==0, 'Missing keys in state_dict: "'+'", "'.join(missing_keys) + '"'


def load_model(model_path, device='cpu'):

	assert os.path.isfile(model_path), "File {} not found.".format(model_path)
	
	# load checkpoint & model info
	ckpt = torch.load(model_path, 'cpu')
	text_encoder_name = ckpt['args'].text_encoder_name
	tokenizer_name = "distilbertUncased" if text_encoder_name=="posetext" else get_tokenizer_name(text_encoder_name)

	# load model
	if ckpt['args'].model == "PoseEmbroider":
		model = PoseEmbroider(
					latentD = ckpt['args'].latentD,
					l2normalize = ckpt['args'].l2normalize,
					num_body_joints = ckpt['args'].num_body_joints,
					text_encoder_name = ckpt['args'].text_encoder_name,
					pose_encoder_name = ckpt['args'].pose_encoder_name,
					image_encoder_name = ckpt['args'].image_encoder_name,
					encoder_projection_type = ckpt['args'].encoder_projection_type,
					external_encoder_projection_type = ckpt['args'].external_encoder_projection_type,
					embroider_core_type = ckpt['args'].embroider_core_type,
					no_projection_heads = ckpt['args'].no_projection_heads
				).to(device)
		
	elif ckpt['args'].model == "Aligner":
		model = Aligner(
					latentD = ckpt['args'].latentD,
					l2normalize = ckpt['args'].l2normalize,
					num_body_joints = ckpt['args'].num_body_joints,
					text_encoder_name = ckpt['args'].text_encoder_name,
					pose_encoder_name = ckpt['args'].pose_encoder_name,
					image_encoder_name = ckpt['args'].image_encoder_name,
					encoder_projection_type = ckpt['args'].encoder_projection_type,
					external_encoder_projection_type = ckpt['args'].external_encoder_projection_type,
				).to(device)
	
	else:
		raise NotImplementedError
	
	# model.load_state_dict(ckpt['model'])
	carefully_load_state_dict(model, ckpt['model'])
	model.eval()
	print(f"Loaded model from (epoch {ckpt['epoch']}):", model_path)

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
		img_processing_scheme = utils.get_img_processing_scheme(model.image_encoder_name)
		# load data
		if "bedlamscript" in dataset_version:
			fs_size = {"val":10000, "train":50000}[split]
			d = BEDLAMScript(version=dataset_version, split=split, tokenizer_name=tokenizer_name, text_index=text_ind, img_processing_scheme=img_processing_scheme, num_body_joints=model.pose_encoder.num_body_joints, cache=True, reduced_set=fs_size, human_visibility=human_visibility)
		elif "threedpw" in dataset_version:
			d = ThreeDPW(version=dataset_version, split=split, img_processing_scheme=img_processing_scheme, num_body_joints=model.pose_encoder.num_body_joints, item_format='ip', cache=True)
		elif "posescript" in dataset_version:
			d = PoseScript(version=dataset_version, split=split, tokenizer_name=tokenizer_name, text_index=text_ind, num_body_joints=model.pose_encoder.num_body_joints, item_format='pt', cache=True)
		else:
			raise NotImplementedError
		# evaluate
		results = compute_eval_metrics(model, d, device)
		# save results
		evaluator.save_results_to_file(results, result_filepath)
	else:
		results = evaluator.load_results_from_file(result_filepath)

	return {k:[v] for k, v in results.items()}


def compute_eval_metrics(model, dataset, device):

	results = {}
	
	# Compute collection features (global modality features)
	collection_features = infer_collection_features(model, dataset, device)

	# Get the subset of available modalities from the dataset setting
	available_modalities = []
	if 'i' in dataset.item_format: available_modalities += ['image']
	if 'p' in dataset.item_format: available_modalities += ['pose']
	if 't' in dataset.item_format: available_modalities += ['text']

	# Compute query features, then matching metrics
	if model.__class__.__name__ == "PoseEmbroider":
		
		# get intermodality token for all input kinds
		intermodality_tokens = infer_intermodality_tokens_for_all_query_types(model, dataset, device)
		
		# compute recalls
		recalls = {}
		# all recall: m_i <--> m_j with i != j
		for m1 in available_modalities:
			query_features = model.get_modality_projections(intermodality_tokens[m1])
			for m2 in available_modalities:
				if m1 != m2:
					r = evaluator.x2y_recall_metrics(query_features[f'predicted_{m2}'],
									 				collection_features[m2],
													config.k_recall_values,
													sstr=f"{m1}-{m2}_")
					recalls.update(r)
		# all recalls: (m_i, m_j) --> m_k with i,j,k all distincts
		if len(available_modalities) > 2:
			for m_target in available_modalities:
				m_query = [m for m in available_modalities if m!=m_target]
				query_features = eval(f'model.modality_projection_{m_target}')(intermodality_tokens['_'.join(m_query)])
				r = evaluator.x2y_recall_metrics(query_features,
									   			collection_features[m_target],
												config.k_recall_values,
												sstr=f"{m_query[0]}+{m_query[1]}2{m_target}_")
				recalls.update(r)

		del collection_features
		del query_features
		del intermodality_tokens

	elif model.__class__.__name__ == "Aligner":
		recalls = {}
		# all recall: m_i <--> m_j with i != j
		for m1 in available_modalities:
			for m2 in available_modalities:
				if m1 != m2:
					r = evaluator.x2y_recall_metrics(collection_features[m1],
									 				collection_features[m2],
													config.k_recall_values,
													sstr=f"{m1}-{m2}_")
					recalls.update(r)
		# all recalls: (m_i, m_j) --> m_k with i,j,k all distincts
		if len(available_modalities) > 2:
			for m_target in available_modalities:
				m_query = [m for m in available_modalities if m!=m_target]
				query_features = model.get_query_features_from_precomputed_features(collection_features, m_query)
				r = evaluator.x2y_recall_metrics(query_features,
									   			collection_features[m_target],
												config.k_recall_values,
												sstr=f"{m_query[0]}+{m_query[1]}2{m_target}_")
				recalls.update(r)
	
	# gather metrics
	r_summarize = {} 
	r_summarize["mRecall_single_query"] = [v for k,v in recalls.items() if '-' in k]
	r_summarize["mRecall_dual_query"] = [v for k,v in recalls.items() if '+' in k]
	r_summarize = {k:sum(v)/len(v) if len(v) else 0 for k,v in r_summarize.items()}
	r_summarize["mRecall"] = (r_summarize["mRecall_single_query"] + r_summarize["mRecall_dual_query"])/2
	
	results.update(r_summarize)
	results.update(recalls)

	return results


def infer_collection_features(model, dataset, device, item_format=None):

	# if the input `item_format` does not inform about the available modalities,
	# deduce the info directly from the dataset setting
	if item_format is None: item_format = dataset.item_format

	# setup dataloader
	batch_size = 32
	data_loader = torch.utils.data.DataLoader(
		dataset, sampler=None, shuffle=False,
		batch_size=batch_size,
		num_workers=8,
		pin_memory=True,
		drop_last=False
	)

	# setup the module to perform data processing (in particular including image
	# processing)
	data_processing = DataProcessingModule(phase="eval",
										nb_joints=model.pose_encoder.num_body_joints,
										img_processing_scheme=dataset.img_processing_scheme)

	# init feature holders
	shape = (len(dataset), model.latentD)
	images_features = torch.zeros(shape).to(device) if "i" in item_format else None
	poses_features = torch.zeros(shape).to(device) if "p" in item_format else None
	texts_features = torch.zeros(shape).to(device) if "t" in item_format else None

	with torch.no_grad() and torch.inference_mode():

		for i, item in tqdm(enumerate(data_loader)):
			
			# load & prepare data
			nb_of_elements = len(item["indices"])
			if 't' in item_format:
				item["texts_tokens"] = item["texts_tokens"][:,:max(item["texts_lengths"])] # truncate within the batch, based on the longest text 
			if 't' not in item_format and 'texts_tokens' in item: del item['texts_tokens']
			if 'i' not in item_format and 'images' in item: del item['images']
			item = data_processing(item) # process images
			item = {k:v.to(device) if k not in ["indices", "dataset"] else v for k,v in item.items() } # set on device
				
			# forward pass
			x = model.get_modality_global_features(**item)
			
			# store computed features
			s = slice(i*batch_size, i*batch_size+nb_of_elements)

			if "i" in item_format: images_features[s] = x['image_emb']
			if "p" in item_format: poses_features[s] = x['pose_emb']
			if "t" in item_format: texts_features[s] = x['text_emb']

	return {'image':images_features,
			'pose':poses_features,
			'text':texts_features}


def infer_intermodality_tokens_for_all_query_types(model, dataset, device):

	# setup dataloader
	batch_size = 32
	data_loader = torch.utils.data.DataLoader(
		dataset, sampler=None, shuffle=False,
		batch_size=batch_size,
		num_workers=8,
		pin_memory=True,
		drop_last=False
	)

	# setup the module to perform data processing (in particular including image
	# processing)
	data_processing = DataProcessingModule(phase="eval",
										nb_joints=model.pose_encoder.num_body_joints,
										img_processing_scheme=dataset.img_processing_scheme)

	# init feature holders
	is_in_item_format = lambda x: x in dataset.item_format
	shape = (len(dataset), model.latentD) # intermodality token
	images_features = torch.zeros(shape).to(device) if is_in_item_format("i") else None
	poses_features = torch.zeros(shape).to(device) if is_in_item_format("p") else None
	texts_features = torch.zeros(shape).to(device) if is_in_item_format("t") else None
	images_poses_features = torch.zeros(shape).to(device) if is_in_item_format("ip") else None
	poses_texts_features = torch.zeros(shape).to(device) if is_in_item_format("pt") else None
	images_texts_features = torch.zeros(shape).to(device) if is_in_item_format("i") and is_in_item_format("t") else None

	with torch.no_grad() and torch.inference_mode():

		for i, item in tqdm(enumerate(data_loader)):
			
			# load & prepare data
			nb_of_elements = len(item["indices"])
			if 't' in dataset.item_format:
				item["texts_tokens"] = item["texts_tokens"][:,:max(item["texts_lengths"])] # truncate within the batch, based on the longest text 
			item = data_processing(item) # process images
			item = {k:v.to(device) if k not in ["indices", "dataset"] else v for k,v in item.items() } # set on device
				
			# forward pass
			x = model.get_intermodality_tokens_for_all_query_types(**item)
			
			# store computed features
			s = slice(i*batch_size, i*batch_size+nb_of_elements)

			if is_in_item_format("i"): images_features[s] = x['images_input']
			if is_in_item_format("p"): poses_features[s] = x['poses_input']
			if is_in_item_format("t"): texts_features[s] = x['texts_input']
			if is_in_item_format("ip"): images_poses_features[s] = x['images_poses_input']
			if is_in_item_format("pt"): poses_texts_features[s] = x['poses_texts_input']
			if is_in_item_format("i") and is_in_item_format("t"): images_texts_features[s] = x['images_texts_input']

	return {'image':images_features,
		 	'pose':poses_features,
			'text':texts_features,
			'image_pose':images_poses_features,
			'pose_text':poses_texts_features,
			'image_text':images_texts_features}


def display_results(results):

	# average recall over the different values of K, for each query type
	avg_over_k = lambda q: [sum([results.get(f'{q}{k}', [0])[0] for k in config.k_recall_values])/len(config.k_recall_values)]
	single_query = ["%s-%s_R@"%(m1,m2) for m1 in ['image', 'pose', 'text'] for m2 in ['image', 'pose', 'text'] if m1!=m2]
	dual_query = ['image+pose2text_R@', 'pose+text2image_R@', 'image+text2pose_R@']
	metric_order = ['mRecall', 'mRecall_single_query', 'mRecall_dual_query'] + single_query + dual_query
	for q in single_query+dual_query:
		results[q] = avg_over_k(q)
	
	results = evaluator.scale_and_format_results(results)
	print("Metric order: " + ' & '.join(metric_order))
	print(f"\n<model> & {' & '.join([results[m] for m in metric_order])} \\\\\n")


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