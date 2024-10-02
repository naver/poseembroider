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
import roma
from functools import reduce
import sys
import os
os.umask(0x0002)
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import text2pose.utils_logging as logging

import poseembroider.utils as utils
from poseembroider.option import get_args_parser
from poseembroider.trainer import GenericTrainer
from poseembroider.datasets.posescript import PoseScript
from poseembroider.datasets.bedlam_script import BEDLAMScript
from poseembroider.augmentations import DataProcessingModule
from poseembroider.assisting_models.poseVAE.model_poseVAE import PoseVAE
from poseembroider.assisting_models.poseVAE.fid import FID
from poseembroider.assisting_models.poseVAE.loss import laplacian_nll, gaussian_nll


class PoseVAETrainer(GenericTrainer):

	def __init__(self, args):
		super(PoseVAETrainer, self).__init__(args)


	def load_dataset(self, split, caption_index, tokenizer_name=None):
		
		data_size = self.args.data_size if split=="train" else None

		if "posescript" in self.args.dataset:
			d = PoseScript(version=self.args.dataset,
				  			split=split,
							num_body_joints=self.args.num_body_joints,
							tokenizer_name=None)
		elif "bedlamscript" in self.args.dataset:
			fs_size = {"val":10000, "train":50000}[split]
			d = BEDLAMScript(version=self.args.dataset,
							split=split,
							num_body_joints=self.args.num_body_joints,
							tokenizer_name=None,
							reduced_set=fs_size,
							item_format="p")
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
		self.model = PoseVAE(latentD=self.args.latentD,
							 num_body_joints=self.args.num_body_joints)
		self.model.to(self.device)


	def get_param_groups(self):
		param_groups = []
		param_groups.append({'params': self.model.pose_encoder.parameters(), 'lr': self.args.lr})
		param_groups.append({'params': self.model.pose_decoder.parameters(), 'lr': self.args.lr})
		param_groups.append({'params': [self.model.decsigma_v2v, self.model.decsigma_jts, self.model.decsigma_rot]})
		return param_groups


	def init_other_training_elements(self):
		self.init_fid(name_in_batch="poses")
		self.init_body_model()
		self.data_processing_module_train = DataProcessingModule(
									phase="train",
									nb_joints=self.args.num_body_joints,
									lr_flip_proba=0.5 if self.args.apply_LR_augmentation else 0)
		self.data_processing_module_val = DataProcessingModule(
									phase="eval",
									nb_joints=self.args.num_body_joints)


	def init_fid(self, name_in_batch):
		# prepare fid on the val set
		if self.args.fid:
			print('Preparing FID.')
			self.fid = FID(self.args.fid, device=self.device, name_in_batch=name_in_batch)
			self.fid.extract_real_features(self.data_loader_val)
		else:
			print('No feature extractor provided to compute the FID. Ignoring FID...')


	def training_epoch(self, epoch):
		train_stats = self.one_epoch(epoch=epoch, is_training=True)
		return train_stats
	

	def validation_epoch(self, epoch):
		if self.args.val_every and (epoch+1)%self.args.val_every==0:
			return self.one_epoch(epoch=epoch, is_training=False)
		return {}
	

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
			if self.args.fid: self.fid.reset_gen_features()

		# iterate over the batches
		for data_iter_step, item in enumerate(metric_logger.log_every(data_loader, self.args.log_step, header)):

			# data augmentations
			if is_training:
				item = self.data_processing_module_train(item)
			else:
				item = self.data_processing_module_val(item)
			
			# get data
			poses = item['poses'].to(self.device)
			
			# forward
			with torch.set_grad_enabled(is_training):
				output = self.model(poses)

			# --- loss computation

			# get body poses for loss computations
			with torch.set_grad_enabled(is_training):
				bm_rec = self.body_model(**utils.pose_data_as_dict(output['pose_body_pose'], code_base="smplx"))
			with torch.no_grad():
				bm_orig = self.body_model(**utils.pose_data_as_dict(poses, code_base="smplx"))

			# (term initialization)
			bs = poses.size(0)
			losses = {k: torch.zeros(bs) for k in ['v2v', 'jts', 'rot', 'kldnp', 'kldnp_training']}

			# (normalization terms: number of coefficients)
			prod = lambda li: reduce(lambda x, y: x*y, li, 1)
			v2v_reweight, jts_reweight, rot_reweight  = [prod(s.shape[1:]) for s in [bm_orig.vertices, bm_orig.joints, output[f'pose_body_matrot_pose']]]

			# (reconstruction terms)
			losses[f'v2v'] = torch.sum(laplacian_nll(bm_orig.vertices, bm_rec.vertices, self.model.decsigma_v2v), dim=[1,2]) # size (batch_size)
			losses[f'jts'] = torch.sum(laplacian_nll(bm_orig.joints, bm_rec.joints, self.model.decsigma_jts), dim=[1,2]) # size (batch_size)
			losses[f'rot'] = torch.sum(gaussian_nll(output[f'pose_body_matrot_pose'].view(-1,self.args.num_body_joints,3,3), roma.rotvec_to_rotmat(poses.view(-1,self.args.num_body_joints,3)), self.model.decsigma_rot), dim=[1,2,3]) # size (batch_size)
			
			# (KL regularization term)
			bs = poses.size(0)
			n_z = torch.distributions.normal.Normal(
				loc=torch.zeros((bs, self.model.latentD), device=self.device, requires_grad=False),
				scale=torch.ones((bs, self.model.latentD), device=self.device, requires_grad=False))
			losses['kldnp'] = torch.sum(torch.distributions.kl.kl_divergence(output['q_z'], n_z), dim=[1]) if self.args.wloss_kldnpmul else torch.tensor(0.0).to(self.device) # size (batch_size)
			losses['kldnp_training'] = losses['kldnp'].clamp(min=self.args.kld_epsilon) if self.args.kld_epsilon else losses['kldnp'] # size (batch_size)
			# NOTE: `kldnp_training` is only to be used at training time;
			# the clamping helps to prevent model collapse
			
			# (total loss)
			wloss_kld = self.args.wloss_kld if is_training else 1.0
			kld_loss = losses['kldnp_training'] if is_training else losses['kldnp']

			loss = self.args.wloss_v2v * (losses['v2v'] + wloss_kld * kld_loss) / v2v_reweight + \
					self.args.wloss_jts * (losses['jts'] + wloss_kld * kld_loss) / jts_reweight + \
					self.args.wloss_rot * (losses['rot'] + wloss_kld * kld_loss) / rot_reweight
			loss = torch.mean(loss)
			
			# sanity check
			loss_value = loss.item()
			if not math.isfinite(loss_value):
				print("Loss is {}, stopping training".format(loss_value))
				sys.exit(1)

			# (prepare loss terms for logging)
			for k, v in losses.items():
				losses[k] = torch.mean(v)

			# --- other computations (elbo, fid...)

			# (elbos)
			# normalization is a bit different than for the losses
			elbos = {}
			elbos['v2v'] = (-torch.sum(laplacian_nll(bm_orig.vertices, bm_rec.vertices, self.model.decsigma_v2v), dim=[1,2]) - losses['kldnp']).sum().detach().item() # (batch_size, nb_vertices, 3): first sum over the coeffs, substract the kld, then sum over the batch
			elbos['jts'] = (-torch.sum(laplacian_nll(bm_orig.joints, bm_rec.joints, self.model.decsigma_jts), dim=[1,2]) - losses['kldnp']).sum().detach().item() # (batch_size, nb_joints, 3): first sum over the coeffs, substract the kld, then sum over the batch
			elbos['rot'] = (-torch.sum(gaussian_nll(output['pose_body_matrot_pose'].view(-1,3,3), roma.rotvec_to_rotmat(poses.view(-1,3)), self.model.decsigma_rot).view(bs, -1, 3, 3), dim =[1,2,3]) - losses['kldnp']).sum().detach().item() # (batch_size, nb_joints, 3, 3): first sum over the coeffs, substract the kld, then sum over the batch
			# normalize, by the batch size and the number of coeffs
			elbos = {
				'v2v': elbos['v2v'] / (bs * v2v_reweight),
				'jts': elbos['jts'] / (bs * jts_reweight),
				'rot': elbos['rot'] / (bs * rot_reweight)}

			# (fid)
			if not is_training and self.args.fid:
				self.fid.add_gen_features( output['pose_body_pose'] )

			# --- training step

			if is_training:
				self.optimizer.zero_grad()
				loss.backward()
				self.optimizer.step()

			# --- logging

			# format data for logging
			scalars = [('loss', loss_value), ('loss', losses), ('elbo', elbos)]
			if is_training:
				lr_value = self.optimizer.param_groups[0]["lr"]
				decsigma = {k:v for k,v in zip(['v2v', 'jts', 'rot'], [self.model.decsigma_v2v, self.model.decsigma_jts, self.model.decsigma_rot])}
				scalars += [('lr', lr_value), ('decsigma', decsigma)]

			# actually log
			self.add_data_to_log_writer(epoch, sstr, scalars=scalars, is_training=is_training, data_iter_step=data_iter_step, total_steps=len(data_loader))
			self.add_data_to_metric_logger(metric_logger, sstr, scalars)


		# computations at the end of the epoch

		# (fid) - must wait to have computed all the features
		if not is_training and self.args.fid:
			scalars = [(self.fid.sstr(), self.fid.compute())]
			metric_logger.add_meter(f'{sstr}_{self.fid.sstr()}', logging.SmoothedValue(window_size=1, fmt='{value:.3f}'))
			self.add_data_to_log_writer(epoch, sstr, scalars=scalars, is_training=is_training, data_iter_step=data_iter_step, total_steps=len(data_loader))
			self.add_data_to_metric_logger(metric_logger, sstr, scalars)


		print("Averaged stats:", metric_logger)
		return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


if __name__ == '__main__':
	
	argparser = get_args_parser()
	args = argparser.parse_args()
	
	PoseVAETrainer(args)()