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

import text2pose.utils_logging as logging
from text2pose.encoders.tokenizers import get_tokenizer_name, Tokenizer

import poseembroider.config as config
import poseembroider.utils as utils
from poseembroider.option import get_args_parser
from poseembroider.trainer import GenericTrainer
from poseembroider.datasets.bedlam_script import BEDLAMScript
from poseembroider.augmentations import DataProcessingModule
from poseembroider.downstream.pose_estimation.model_pose_estimation import HPSEstimator
from poseembroider.downstream.pose_estimation.evaluate_pose_estimation import compute_eval_metrics


################################################################################

class HPSEstimatorTrainer(GenericTrainer):

	def __init__(self, args):
		super(HPSEstimatorTrainer, self).__init__(args)
		# switch setup to the generic case where multiple datasets can be processed at once
		if not self.args.datasets: self.args.datasets = [self.args.dataset]


	def get_tokenizer_name(self):
		return "distilbertUncased" if self.representation_model_config['text_encoder_name']=="posetext" else get_tokenizer_name(self.representation_model_config['text_encoder_name'])


	def get_item_format(self):
		# define what kind of data to load
		item_format = ''
		if 'images' in self.args.representation_model_input_types and not self.args.cached_embeddings_file:
			item_format += 'i' # do not load images, we already have their features from the cache file
		if 'texts' in self.args.representation_model_input_types and not self.args.cached_embeddings_file:
			item_format += 't' # do not load texts, we already have their features from the cache file
		item_format += 'p' # of course (that's the target, we need it to compute the loss)
		return sorted(item_format) # alphabetic order


	def load_sub_dataset(self, dataset_version, split, caption_index, tokenizer_name=None):

		# define tokenizer		
		if tokenizer_name is None: tokenizer_name = self.get_tokenizer_name()
		
		# define image processing scheme
		self.img_processing_scheme = utils.get_img_processing_scheme(self.representation_model_config['image_encoder_name'])

		# define & load dataset
		if 'bedlamscript' in dataset_version:
			fs_size = {"val":10000, "train":50000}[split]
			d = BEDLAMScript(version=dataset_version,
							split=split,
							item_format=self.get_item_format(),
							tokenizer_name=tokenizer_name,
							text_index=caption_index,
							img_processing_scheme=self.img_processing_scheme,
							num_body_joints=self.args.num_body_joints,
							num_shape_coeffs=self.args.num_shape_coeffs,
							reduced_set=fs_size)
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
		
		self.model = HPSEstimator(
				num_body_joints=self.args.num_body_joints,
				num_shape_coeffs=self.args.num_shape_coeffs,
				predict_bodyshape = self.args.predict_bodyshape,
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
				"text_encoder_name": ckpt['args'].text_encoder_name,
				"image_encoder_name": ckpt['args'].image_encoder_name,
			}
		else:
			self.representation_model_config = {
				"text_encoder_name": self.model.representation_wrapper.representation_model.text_encoder_name,
				"image_encoder_name": self.model.representation_wrapper.representation_model.image_encoder_name,
			}


	def get_param_groups(self):
		param_groups = []
		params = [p for p in self.model.pose_prediction_head.parameters() if p.requires_grad]
		if self.args.predict_bodyshape:
			params += [p for p in self.model.bodyshape_prediction_head.parameters() if p.requires_grad]
		# NOTE: do not include the parameters of the representation model (if
		# loaded); its weights are frozen anyway
		param_groups.append({'params': params, 'lr': self.args.lr})
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
		tokenizer_name = self.get_tokenizer_name()
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
		# body model
		self.init_body_model(num_betas=self.args.num_shape_coeffs) # needed for the validation phase, and if self.args.smpl_losses


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

			# data processing
			if not self.args.cached_embeddings_file:
				if 'texts_tokens' in item:
					item["texts_tokens"] = item["texts_tokens"][:,:max(item["texts_lengths"])] # truncate within the batch, based on the longest text 

				# image processing + data augmentations
				if is_training:
					item = self.data_processing_module_train(item)
				else:
					item = self.data_processing_module_val(item)

			# forward + loss term computation
			input_dict = {k:v.to(self.device) if k not in ["indices", "dataset"] else v for k,v in item.items() } # set on device
			with torch.set_grad_enabled(is_training):
				losses = self.model.forward_loss(item=input_dict,
									representation_model_input_types=self.args.representation_model_input_types, 
									pose_pred_smpl_losses=self.args.smpl_losses,
									body_model=self.body_model
								)

			# compute total loss
			loss = sum(losses.values())/len(losses)
			loss_value = loss.item()
			if not math.isfinite(loss_value):
				print("Loss is {}, stopping training.".format(loss_value))
				sys.exit(1)

			# training step
			if is_training:
				self.optimizer.zero_grad()
				loss.backward()
				self.optimizer.step()

			# format data for logging
			scalars = [('loss', loss_value), ('loss', losses)]
			if is_training:
				lr_value = self.optimizer.param_groups[0]["lr"]
				scalars += [('lr', lr_value)]

			# actually log
			self.add_data_to_log_writer(epoch, sstr, scalars=scalars, is_training=is_training, data_iter_step=data_iter_step, total_steps=len(data_loader))
			self.add_data_to_metric_logger(metric_logger, sstr, scalars)

		print("Averaged stats:", metric_logger)
		return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


	def validate(self, epoch, dname):

		self.model.eval()

		# get the right dataset
		if len(self.args.datasets)>1:
			dataset_val = self.data_loader_val.dataset.get_sub_dataset(dname)
		else:
			dataset_val = self.data_loader_val.dataset

		# compute metrics
		val_stats = compute_eval_metrics(self.model,
								   dataset_val,
								   self.device,
								   body_model=self.body_model,
								   representation_model_input_types=self.args.representation_model_input_types,
								   )

		# log
		self.add_data_to_log_writer(epoch, 'val', scalars=[('validation', val_stats)], should_log_data=True)
		print(f"[val] Epoch: [{epoch}] Stats: " + "  ".join(f"{k}: {round(v, 3)}" for k,v in val_stats.items()) )
		return val_stats	


if __name__ == '__main__':
	
	argparser = get_args_parser()
	args = argparser.parse_args()
	
	HPSEstimatorTrainer(args)()