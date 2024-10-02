##############################################################
## poseembroider                                            ##
## Copyright (c) 2024                                       ##
## Institut de Robotica i Informatica Industrial, CSIC-UPC  ##
## and Naver Corporation                                    ##
## Licensed under the CC BY-NC-SA 4.0 license.              ##
## See project root for license details.                    ##
##############################################################

import torch
import math
import sys
import os
os.umask(0x0002)
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

from text2pose.encoders.tokenizers import Tokenizer, get_tokenizer_name
import text2pose.utils_logging as logging

import poseembroider.utils as utils
import poseembroider.config as config
from poseembroider.option import get_args_parser
from poseembroider.trainer import GenericTrainer
from poseembroider.downstream.instruction_generation.model_instruction_generation import InstructionGenerator
from poseembroider.downstream.instruction_generation.evaluate_instruction_generation import get_evaluation_model_paths, get_evaluation_models, compute_eval_metrics
from poseembroider.datasets.bedlam_fix import BEDLAMFix
from poseembroider.datasets.posefix import PoseFix
from poseembroider.augmentations import DataProcessingModule


################################################################################
class InstructionGenerationTrainer(GenericTrainer):

	def __init__(self, args):
		super(InstructionGenerationTrainer, self).__init__(args)
		# switch setup to the generic case where multiple datasets can be processed at once
		if not self.args.datasets: self.args.datasets = [self.args.dataset]
		if not self.args.pair_kinds: self.args.pair_kinds = [self.args.pair_kind]


	def get_item_format(self, split):
		# define what kind of data to load
		item_format = ''
		if 'images' in self.args.representation_model_input_types and not self.args.cached_embeddings_file:
			item_format += 'i' # do not load images, we already have their features from the cache file
		if split=="train":
			if 'poses' in self.args.representation_model_input_types and not self.args.cached_embeddings_file:
				item_format += 'p' # do not load poses, we already have their features from the cache file
		else:
			# NOTE: at validation time, we need 3D poses to use the textret model
			item_format += 'p'
		item_format += 't' # of course (that's the target, we need it to compute the loss)
		return item_format
	

	def load_sub_dataset(self, dataset_version, split, caption_index, tokenizer_name=None):
		
		# define tokenizer
		if tokenizer_name is None: tokenizer_name = get_tokenizer_name(self.args.text_decoder_name)

		# define image processing scheme
		self.img_processing_scheme = utils.get_img_processing_scheme(self.representation_model_config['image_encoder_name'])
	
		# other data properties
		pair_kind = self.args.pair_kinds[self.args.datasets.index(dataset_version)]

		# load dataset
		item_format = self.get_item_format(split)
		if 'posefix' in dataset_version:
			d = PoseFix(version=dataset_version,
			   			split=split,
						tokenizer_name=tokenizer_name,
						text_index=caption_index,
						num_body_joints=self.args.num_body_joints,
						item_format=item_format if item_format[0]!='i' else item_format[1:], # PoseFix does not handle images
						pair_kind=pair_kind)
		elif 'bedlamfix' in dataset_version:
			fs_size = {"val":10000, "train":50000}[split]
			d = BEDLAMFix(version=dataset_version,
				 		split=split,
						item_format=item_format,
						tokenizer_name=tokenizer_name,
						text_index=caption_index,
						img_processing_scheme=self.img_processing_scheme,
						num_body_joints=self.args.num_body_joints,
						reduced_set=fs_size,
						pair_kind=pair_kind)
		else:
			raise NotImplementedError

		# adapt size
		data_size = self.args.data_size if split=="train" else None
		if data_size:
			initial_size = len(d)
			d.index_2_id_list = d.index_2_id_list[:data_size]
			print(f"[Using reduced dataset!] Size: {initial_size} --> {len(d)}")

		return d

	def get_cache_embedding_files(self, split:str, datasets:list) -> dict:
		"""
		Return:
			cache_embeddings_file: dict formatted as {dataset_name:filename}
		"""

		cached_embeddings_files = {}
		filemark = self.args.cached_embeddings_file

		for d_version in datasets:

			# define `precision` for special cases
			if 'bedlamscript' in d_version:
				fs_size = {"val":10000, "train":50000}[split]
				precision = f"_fs{fs_size}"
			else: 
				precision = ""

			# define full cache filename
			cached_embeddings_files[d_version] = f"cached_features_{d_version}_{filemark}_{split}{precision}_input-{'-'.join(self.args.representation_model_input_types)}.pt"

		return cached_embeddings_files


	def init_model(self):
		print('Load model')

		path_to_pretrained_representation_model = utils.read_json(config.PRETRAINED_MODEL_DICT)[self.args.pretrained_representation_model]

		self.model = InstructionGenerator(
								num_body_joints=self.args.num_body_joints,
								comparison_latentD=self.args.comparison_latentD,
								comparison_module_mode=self.args.comparison_module_mode,
								text_decoder_name=self.args.text_decoder_name,
								transformer_mode=self.args.transformer_mode,
								decoder_latentD=self.args.decoder_latentD,
								decoder_nhead=self.args.decoder_nhead,
								decoder_nlayers=self.args.decoder_nlayers,
								# -- about the representation model
								encoder_latentD=self.args.latentD,
								path_to_pretrained_representation_model=path_to_pretrained_representation_model,
								cached_embeddings_file=self.get_cache_embedding_files(split="train", datasets=self.args.datasets) if self.args.cached_embeddings_file else False,
								)								
		self.model.to(self.device)

		# get config info
		if self.args.cached_embeddings_file:
			# the representation model has not been loaded: need to load
			# the model temporarily to get the proper config
			ckpt = torch.load(path_to_pretrained_representation_model, 'cpu')
			self.representation_model_config = {
				"image_encoder_name": ckpt['args'].image_encoder_name,
			}
		else:
			self.representation_model_config = {
				"image_encoder_name": self.model.representation_wrapper.representation_model.image_encoder_name,
			}

		# convert general input types to role-related input type
		# ie. ["poses"] --> ["poses_A", "poses_B"]
		self.actual_representation_model_input_types = []
		for it in self.args.representation_model_input_types:
			self.actual_representation_model_input_types += [f"{it}_A", f"{it}_B"]


	def get_param_groups(self):
		param_groups = []
		param_groups.append({'params': [p for p in self.model.comparison_module.parameters() if p.requires_grad] + \
			  						[p for p in self.model.modality_input_adapter.parameters() if p.requires_grad] + \
					   				[p for p in self.model.text_decoder.parameters() if p.requires_grad],
									'lr': self.args.lr})
		# NOTE: do not include the parameters of the representation model (if
		# loaded); its weights are frozen anyway
		return param_groups


	def carefully_load_ckpt(self, ckpt):

		def filter_out_representation_model_keys(keys, key_type):
			actual_keys = [k for k in keys if not k.startswith("representation_wrapper.representation_model.")]
			assert len(actual_keys) == 0, f"{key_type.capitalize()} weight keys when loading the model checkpoint."

		missing_keys, unexpected_keys = self.model.load_state_dict(ckpt['model'], strict=False)
		if self.args.cached_embeddings_file and len(unexpected_keys):
			# Ignore unexpected keys related to the 'representation_model', as
			# the use of the cached features resulted in the
			# representation_model not being initialized.
			# Ensure these are the only unexpected keys.
			filter_out_representation_model_keys(unexpected_keys, "unexpected")
		elif len(missing_keys):
			# Ignore missing keys related to the 'representation_model',
			# assuming the reason for this is that the previous training made
			# use of the cached features.
			# Ensure these are the only keys missing.
			filter_out_representation_model_keys(missing_keys, "missing")


	def start_or_resume_training(self):
		if os.path.isfile(self.ckpt_fname): # resume training
			print("Resume training. Load weights from last checkpoint.")
			ckpt = torch.load(self.ckpt_fname, 'cpu')
			self.start_epoch = ckpt['epoch'] + 1
			self.best_val_loss = ckpt.get('best_val_loss', float('inf'))
			self.carefully_load_ckpt(ckpt)
			self.optimizer.load_state_dict(ckpt['optimizer'])
			if self.lr_scheduler:
				self.lr_scheduler.load_state_dict(ckpt['scheduler'])
		else:
			self.start_epoch = 0
			self.best_val_loss = float('inf')
			if self.args.pretrained: # load pretrained weights
				pretrained_path = utils.read_json(config.PRETRAINED_MODEL_DICT)[self.args.pretrained]
				print(f'Loading pretrained model: {pretrained_path}')
				ckpt = torch.load(pretrained_path, 'cpu')
				assert self.args.num_body_joints == getattr(ckpt['args'], 'num_body_joints', 52), "Pose-related modules in the initialized model and the pretrained model use a different number of joints."
				self.carefully_load_ckpt(ckpt)


	def init_other_training_elements(self):
		# data processing & augmentations
		tokenizer_name = get_tokenizer_name(self.args.text_decoder_name)
		tokenizer = Tokenizer(tokenizer_name)
		self.data_processing_module_train = DataProcessingModule(
											phase="train",
											nb_joints=self.args.num_body_joints,
											lr_flip_proba=0.5 if self.args.apply_LR_augmentation else 0,
											img_processing_scheme=self.img_processing_scheme,
											no_img_augmentations=args.no_img_augmentation,
											tokenizer=tokenizer,
											)
		self.data_processing_module_val = DataProcessingModule(
											phase="eval",
											nb_joints=self.args.num_body_joints,
											img_processing_scheme=self.img_processing_scheme
											)


	def training_epoch(self, epoch):
		train_stats = self.one_epoch(epoch=epoch, is_training=True)
		return train_stats
	

	def validation_epoch(self, epoch):
		val_stats = {}

		if self.args.val_every and (epoch+1)%self.args.val_every==0:

			# adapt cached features to validation data features
			if self.args.cached_embeddings_file:
				val_cached_embeddings_file = self.get_cache_embedding_files(split='val', datasets=self.args.datasets)
				self.model.representation_wrapper.load_cache_features(val_cached_embeddings_file)

			# loss-kind validation step
			# (NOTE: using the dataloader with mixed datasets, if several)
			val_stats.update(self.one_epoch(epoch=epoch, is_training=False))

			# metric-kind validation step: run dataset-specific validation
			if self.args.val_every and (epoch+1)%(self.args.val_every*2)==0:
				for dname in self.args.datasets:
					val_dname = self.validate(epoch=epoch, dname=dname)
					val_stats.update({f'{dname}_{k}':v for k,v in val_dname.items()})

			# revert cached features to training data features
			if self.args.cached_embeddings_file:
				train_cached_embeddings_file = self.get_cache_embedding_files(split="train", datasets=self.args.datasets)
				self.model.representation_wrapper.load_cache_features(train_cached_embeddings_file)

		return val_stats
	

	def one_epoch(self, epoch, is_training):

		self.model.train(is_training)

		# define loggers
		metric_logger = logging.MetricLogger(delimiter="  ")
		if is_training:
			prefix, sstr = '', 'train'
			metric_logger.add_meter(f'{sstr}_lr', logging.SmoothedValue(window_size=1, fmt='{value:.6f}'))
		else:
			prefix, sstr = '[val] ', 'val'
		header = f'{prefix}Epoch: [{epoch}]'
		
		# define dataloader & other elements
		if is_training:
			data_loader = self.data_loader_train
		if not is_training:
			data_loader = self.data_loader_val

		# iterate over the batches
		for data_iter_step, item in enumerate(metric_logger.log_every(data_loader, self.args.log_step, header)):

			# get data
			item["texts_tokens"] = item["texts_tokens"][:,:max(item["texts_lengths"])] # truncate within the batch, based on the longest text 

			# image processing + data augmentations
			if not self.args.cached_embeddings_file:
				if is_training:
					item = self.data_processing_module_train(item)
				else:
					item = self.data_processing_module_val(item)

			# forward + compute loss
			input_dict = {k:v.to(self.device) if k not in ["indices", "dataset"] else v for k,v in item.items() }
			with torch.set_grad_enabled(is_training):
				output = self.model(item=input_dict,
									representation_model_input_types=self.actual_representation_model_input_types)

			loss = output["loss"]
			loss_value = loss.item()
			if not math.isfinite(loss_value):
				print("Loss is {}, stopping training".format(loss_value))
				sys.exit(1)

			# training step
			if is_training:
				self.optimizer.zero_grad()
				loss.backward()
				self.optimizer.step()
				
			# format data for logging
			scalars = [('loss', loss_value)]
			if is_training:
				lr_value = self.optimizer.param_groups[0]["lr"]
				scalars += [('lr', lr_value)]
			else: # validation
				scalars += [('fake_loss', output['fake_loss'])]
				
			# actually log
			self.add_data_to_log_writer(epoch, sstr, scalars=scalars, is_training=is_training, data_iter_step=data_iter_step, total_steps=len(data_loader))
			self.add_data_to_metric_logger(metric_logger, sstr, scalars)

		print("Averaged stats:", metric_logger)
		return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


	def validate(self, epoch, dname):

		self.model.eval()

		# load evaluation models
		textret_model_path, _ = get_evaluation_model_paths(textret_model_version=self.args.textret_model)
		textret_model, tokenizer_name_textret_model = get_evaluation_models(self.device, textret_model_path)

		# get dataset
		if len(self.args.datasets)>1:
			dataset_val = self.data_loader_val.dataset.get_sub_dataset(dname)
		else:
			dataset_val = self.data_loader_val.dataset

		# compute metrics
		metrics = compute_eval_metrics(self.model,
								dataset_val,
								self.device,
								representation_model_input_types=self.actual_representation_model_input_types,
								textret_model=textret_model,
								tokenizer_name_textret_model=tokenizer_name_textret_model)

		# log
		self.add_data_to_log_writer(epoch, 'val', scalars=[('validation', metrics)], should_log_data=True)
		print(f"[val] Epoch: [{epoch}][{dname}] Stats: " + "  ".join(f"{k}: {round(v, 3)}" for k,v in metrics.items()) )
		return metrics		


if __name__ == '__main__':
	
	argparser = get_args_parser()
	args = argparser.parse_args()
	
	InstructionGenerationTrainer(args)()