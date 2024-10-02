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

from text2pose.encoders.tokenizers import get_tokenizer_name, Tokenizer
import text2pose.utils_logging as logging

import poseembroider.utils as utils
from poseembroider.option import get_args_parser
from poseembroider.trainer import GenericTrainer
from poseembroider.model import PoseEmbroider, Aligner
from poseembroider.evaluate_core import compute_eval_metrics
from poseembroider.datasets.bedlam_script import BEDLAMScript
from poseembroider.augmentations import DataProcessingModule


class PoseEmbroiderTrainer(GenericTrainer):

	def __init__(self, args):
		super(PoseEmbroiderTrainer, self).__init__(args, retrieval_trainer=True)


	def get_tokenizer_name(self):
		return "distilbertUncased" if self.args.text_encoder_name=="posetext" else get_tokenizer_name(self.args.text_encoder_name)


	def load_dataset(self, split, caption_index, tokenizer_name=None):

		# define tokenizer		
		if tokenizer_name is None: tokenizer_name = self.get_tokenizer_name()
		# define image processing scheme
		self.img_processing_scheme = utils.get_img_processing_scheme(self.args.image_encoder_name)

		# define & load dataset
		if 'bedlamscript' in self.args.dataset:
			fs_size = {"val":10000, "train":50000}[split]
			d = BEDLAMScript(version=self.args.dataset,
								split=split,
								tokenizer_name=tokenizer_name,
								text_index=caption_index,
								img_processing_scheme=self.img_processing_scheme,
								num_body_joints=self.args.num_body_joints,
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


	def init_model(self):
		print('Load model')
		if args.model == "PoseEmbroider":
			self.model = PoseEmbroider(
					latentD = self.args.latentD,
					l2normalize = self.args.l2normalize,
					num_body_joints = self.args.num_body_joints,
					text_encoder_name = self.args.text_encoder_name,
					pose_encoder_name = self.args.pose_encoder_name,
					image_encoder_name = self.args.image_encoder_name,
					encoder_projection_type = self.args.encoder_projection_type,
					external_encoder_projection_type = self.args.external_encoder_projection_type,
					embroider_core_type = self.args.embroider_core_type,
					no_projection_heads = self.args.no_projection_heads
				)
		elif args.model == "Aligner":
			self.model = Aligner(
					latentD = self.args.latentD,
					l2normalize = self.args.l2normalize,
					num_body_joints = self.args.num_body_joints,
					text_encoder_name = self.args.text_encoder_name,
					pose_encoder_name = self.args.pose_encoder_name,
					image_encoder_name = self.args.image_encoder_name,
					encoder_projection_type = self.args.encoder_projection_type,
					external_encoder_projection_type = self.args.external_encoder_projection_type,
				)
		self.model.to(self.device)


	def get_param_groups(self):
		param_groups = [{'params': [p for p in self.model.parameters() if p.requires_grad], 'lr': self.args.lr}]
		return param_groups


	def init_optimizer(self):
		assert self.args.optimizer=='Adam'
		param_groups = self.get_param_groups()
		self.optimizer = torch.optim.Adam(param_groups, lr=self.args.lr)


	def init_lr_scheduler(self):
		self.lr_scheduler = None
		if self.args.lr_scheduler == "stepLR":
			self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,
													step_size=self.args.lr_step,
													gamma=self.args.lr_gamma,
													last_epoch=-1)


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


	def training_epoch(self, epoch):
		train_stats = self.one_epoch(epoch=epoch, is_training=True)
		return train_stats
	

	def validation_epoch(self, epoch):
		val_stats = {}
		if self.args.val_every and (epoch+1)%self.args.val_every==0:
			val_stats.update(self.one_epoch(epoch=epoch, is_training=False))
			val_stats.update(self.validate(epoch=epoch))
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

			# truncate within the batch, based on the longest text
			item["texts_tokens"] = item["texts_tokens"][:,:max(item["texts_lengths"])]

			# image processing + data augmentations
			if is_training:
				item = self.data_processing_module_train(item)
			else:
				item = self.data_processing_module_val(item)

			# set on device
			input_dict = {k:v.to(self.device) if k not in ["indices", "dataset"] else v for k,v in item.items() }

			# forward + loss term computation
			with torch.set_grad_enabled(is_training):
				losses = self.model(**input_dict,
									loss_type=self.args.retrieval_loss,
									single_partials=self.args.single_partials,
									dual_partials=self.args.dual_partials,
									triplet_partial=self.args.triplet_partial
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


	def validate(self, epoch):

		self.model.eval()
		val_stats = compute_eval_metrics(self.model, self.data_loader_val.dataset, self.device)

		# log
		self.add_data_to_log_writer(epoch, 'val', scalars=[('validation', val_stats)], should_log_data=True)
		print(f"[val] Epoch: [{epoch}] Stats: " + "  ".join(f"{k}: {round(v, 3)}" for k,v in val_stats.items()) )
		return val_stats	


if __name__ == '__main__':
	
	argparser = get_args_parser()
	args = argparser.parse_args()
	
	PoseEmbroiderTrainer(args)()