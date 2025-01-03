##############################################################
## poseembroider                                            ##
## Copyright (c) 2024                                       ##
## Institut de Robotica i Informatica Industrial, CSIC-UPC  ##
## and Naver Corporation                                    ##
## Licensed under the CC BY-NC-SA 4.0 license.              ##
## See project root for license details.                    ##
##############################################################

from typing import Union
import time, datetime
import numpy as np
import json
from pathlib import Path
import re
import shutil
import os
os.umask(0x0002)
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import torch
import torch.backends.cudnn as cudnn
# from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid

import poseembroider.config as config
import poseembroider.utils as utils
from poseembroider.option import get_output_dir


################################################################################


class CustomConcatDataset(torch.utils.data.ConcatDataset):
    def __init__(self, datasets:list):
        super().__init__(datasets) # NOTE: this will initialize self.datasets
        self.sub_dataset_lengths = [len(d) for d in datasets] 

    def get_sub_dataset(self, dname):
        """Find the composing dataset with the required name."""
        for d in self.datasets:
            if d.get_effective_version() == dname:
                return d
        return False


class SuperDatasetWeightedRandomSampler(torch.utils.data.WeightedRandomSampler):
    def __init__(self, dataset_lengths: list, balance_weights:Union[list,None]=None):
        """
        Args:
            dataset_lengths: list of the lengths of the datasets composing the
                            super-dataset at stake.
            balance_weights: list of weights defining the relative sampling
                            frequence of each composing dataset. These are
                            understood with respect to the smaller dataset; ie.
                            if the smaller dataset is of size 'n', and a dataset
                            is given weight '2', the sampler will draw 2*n
                            samples from this dataset at each epoch, in average.
                            If None, it is default to all weights equal to 1.

        """
        # get the total size considered to represent one epoch
        if not balance_weights:
            balance_weights = [1 for _ in range(len(dataset_lengths))]
        bw_tot = sum(balance_weights)
        minimal_size = min([dl for dl in dataset_lengths])
        self.balanced_size = int(minimal_size * bw_tot)

        # define the sampling probability of each sample, so that an epoch
        # consists in sampling `w_i*n` samples for dataset `i` where `w_i` is
        # the input balance weight for this dataset and `n` is the size of the
        # smallest dataset  
        samples_weight = torch.concat([
            (bw*minimal_size)/(bw_tot*dl) * torch.ones(dl)
            for (bw, dl) in zip(balance_weights, dataset_lengths)
        ])

        # finally define the original WeightedRandomSampler based on these
        # deduced parameters        
        super().__init__(samples_weight, self.balanced_size)


class GenericTrainer():

	def __init__(self, args, retrieval_trainer=False):
		super(GenericTrainer, self).__init__()

		# define setting
		self.args = args
		self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
		self.retrieval_trainer = retrieval_trainer
		self.the_higher_the_better = retrieval_trainer # NOTE: default, general case, but not systematic!
		self.best_score_metric_name = 'mRecall' if retrieval_trainer else 'val_loss'
		cudnn.benchmark = True

		# fix the seed for reproducibility
		seed = args.seed
		torch.manual_seed(seed)
		np.random.seed(seed)

		# create path to saving directory
		if args.output_dir=='':
			args.output_dir = get_output_dir(args)
		Path(args.output_dir).mkdir(parents=True, exist_ok=True)
		print(args.output_dir)
		# define checkpoint
		self.ckpt_fname = os.path.join(self.args.output_dir, 'checkpoint_last.pth')


	# METHODS TO DEFINE IN CHILD CLASSES
	############################################################################

	def init_model(self):
		"""
		Must define self.model.
		"""
		raise NotImplementedError


	def get_param_groups(self):
		"""
		Must return a list of param groups.
		"""
		raise NotImplementedError


	def init_other_training_elements(self):
		"""
		Must initialize as attributes other elements necessary at training or
		validation time, in particular the data augmentation module. It can be
		used to include the initialization of elements like the FID or the body
		model. This method should not return anything.
		"""
	
	def load_dataset(self, split, caption_index):
		"""
		Must return a dataset.
		"""
		raise NotImplementedError


	def load_sub_dataset(self, dataset_version, split, caption_index, tokenizer_name=None):
		"""
		Must return a dataset.
		"""
		raise NotImplementedError
	

	def training_epoch(self):
		"""
		Must return a dict of training statistics.
		"""
		raise NotImplementedError
	

	def validation_epoch(self, epoch):
		"""
		Must return a dict of validation statistics.
		"""
		raise NotImplementedError


	# GENERIC METHODS
	############################################################################


	# TRAINING STATUS ----------------------------------------------------------


	def check_training_status(self):
		if os.path.isfile(self.ckpt_fname):
			ckpt = torch.load(self.ckpt_fname, 'cpu')
			if ckpt["epoch"] == self.args.epochs - 1:
				print(f'Training already done ({self.args.epochs} epochs: {os.path.basename(self.ckpt_fname)})!')
				return
			

	def start_or_resume_training(self):
		if os.path.isfile(self.ckpt_fname): # resume training
			print("Resume training. Load weights from last checkpoint.")
			ckpt = torch.load(self.ckpt_fname, 'cpu')
			self.start_epoch = ckpt['epoch'] + 1
			if self.retrieval_trainer:
				self.best_score = ckpt.get('best_score', 0.0 if self.the_higher_the_better else float('inf')) # init
			else:
				self.best_val_loss = ckpt.get('best_val_loss', float('inf'))
			self.model.load_state_dict(ckpt['model'])
			self.optimizer.load_state_dict(ckpt['optimizer'])
			if self.lr_scheduler:
				self.lr_scheduler.load_state_dict(ckpt['scheduler'])
		else:
			self.start_epoch = 0
			if self.retrieval_trainer:
				self.best_score = 0.0 if self.the_higher_the_better else float('inf') # init
			else:
				self.best_val_loss = float('inf')
			if self.args.pretrained: # load pretrained weights
				pretrained_path = utils.read_json(config.PRETRAINED_MODEL_DICT)[self.args.pretrained]
				print(f'Loading pretrained model: {pretrained_path}')
				ckpt = torch.load(pretrained_path, 'cpu')
				self.model.load_state_dict(ckpt['model'])


	def has_model_improved(self, val_stats):
		if self.retrieval_trainer:
			if val_stats:
				if (val_stats[self.best_score_metric_name] > self.best_score and self.the_higher_the_better) or \
					(val_stats[self.best_score_metric_name] < self.best_score and not self.the_higher_the_better):
					self.best_score = val_stats[self.best_score_metric_name]
					return True
		else:
			if val_stats and val_stats[self.best_score_metric_name] < self.best_val_loss:
				self.best_val_loss = val_stats[self.best_score_metric_name]
				return True
		return False
	

	def save_checkpoint(self, save_best_model, epoch):
		tosave = {'epoch': epoch,
				'model': self.model.state_dict(),
				'optimizer': self.optimizer.state_dict(),
				'scheduler': self.lr_scheduler.state_dict() if self.lr_scheduler else None,
				'args': self.args}
		if self.retrieval_trainer:
			tosave['best_score'] = self.best_score
		else:
			tosave['best_val_loss'] = self.best_val_loss
		# overwrite the current checkpoint after each epoch
		torch.save(tosave, self.ckpt_fname)
		# overwrite the "best" checkpoint each time the model improves
		if save_best_model:
			shutil.copyfile(self.ckpt_fname, os.path.join(self.args.output_dir, 'checkpoint_best.pth'))
		# save an independent checkpoint every required time step
		if epoch and epoch % self.args.saving_ckpt_step == 0:
			shutil.copyfile(self.ckpt_fname, os.path.join(self.args.output_dir, 'checkpoint_{:d}.pth'.format(epoch)))
		

	# INITIALIZATIONS ----------------------------------------------------------


	def load_datasets(self, split, caption_index):
		"""
		Returns:
			dataset
			sampler|None
		"""
		if self.args.datasets:
			if len(self.args.datasets)>1:
				# load all datasets
				d = CustomConcatDataset([self.load_sub_dataset(dataset_version, split, caption_index) for dataset_version in self.args.datasets])
				# define sampler
				sampler = SuperDatasetWeightedRandomSampler(d.sub_dataset_lengths, self.args.dataset_weight_balance)
				# print information
				print(f"[{split}] Concatenated datasets and sampling in a balanced way with the following weights:\n\t- " + \
						  '\n\t- '.join([f'{dname}: {w}' for (dname, w) in zip(self.args.datasets, self.args.dataset_weight_balance)]) + \
						f"\nNumber of samples for one epoch: {sampler.balanced_size}")
				return d, sampler
			else:
				# there's actually only one dataset
				return self.load_sub_dataset(self.args.datasets[0], split, caption_index), None
		else:
			try:
				return self.load_sub_dataset(self.args.dataset, split, caption_index), None
			except NotImplementedError:
				# print("Support to previous code version: always loading a single dataset!")
				return self.load_dataset(split, caption_index), None


	def init_dataloaders(self):
		print('Load training dataset')
		dataset_train, sampler = self.load_datasets('train', caption_index='rand')
		print(dataset_train, len(dataset_train))
		self.data_loader_train = torch.utils.data.DataLoader(
			dataset_train,
			sampler=sampler, shuffle=not sampler,
			batch_size=self.args.batch_size,
			num_workers=self.args.num_workers,
			pin_memory=True,
			drop_last=True,
		)
		print('Load validation dataset')
		dataset_val, _ = self.load_datasets('val', caption_index='deterministic-mix')
		print(dataset_val, len(dataset_val))
		self.data_loader_val = torch.utils.data.DataLoader(
			dataset_val, sampler=None, shuffle=False,
			batch_size=self.args.batch_size,
			num_workers=self.args.num_workers,
			pin_memory=True,
			drop_last=False
		)


	def init_optimizer(self):
		assert self.args.optimizer=='Adam'
		param_groups = self.get_param_groups()
		self.optimizer = torch.optim.Adam(param_groups, lr=self.args.lr, weight_decay=self.args.wd)


	def init_lr_scheduler(self):
		self.lr_scheduler = None


	# (other optional initializations) -----------------------------------------
	# (to be added, if need be, in the overriden version of
	# self.init_other_training_elements)


	def init_body_model(self, num_betas = config.NB_SHAPE_COEFFS):
		print(f'Initialize body model')
		self.body_model = utils.BodyModelSMPLX(config.SMPLX_BODY_MODEL_PATH, batch_size=self.args.batch_size, num_betas=num_betas)
		self.body_model.eval()
		self.body_model.to(self.device)


	# TRAINING PROCESS ---------------------------------------------------------


	def print_trainable_params(self):
		
		optimized_params = self.get_param_groups() # list of dicts {'params':..., 'lr':...}
		optimized_params = [v['params'] for v in optimized_params] # list of lists
		optimized_params = [element for sublist in optimized_params for element in sublist] # single list
		
		requires_grad_params = [] # params whose gradient will be recorded
		optimized_param_names = [] # params that will be optimized
		for n, p in self.model.named_parameters():
			if p.requires_grad == True:
				requires_grad_params.append(n)
			for op in optimized_params:
				if p is op:
					optimized_param_names.append(n)
					break
		
		# params that will be updated
		requires_grad_params = set(requires_grad_params)
		optimized_param_names = set(optimized_param_names)
		print("Trainable parameters:", sorted(requires_grad_params.intersection(optimized_param_names)))
		t = requires_grad_params.difference(optimized_param_names)
		if t: print("### WARNING ### - the following parameters have requires_grad=True, but won't be optimized:", sorted(t))
		t = optimized_param_names.difference(requires_grad_params)
		if t: print("### WARNING ### - the following parameters will be 'optimized', but their gradient is not recorded:", sorted(t))


	def add_data_to_log_writer(self, epoch, sstr, scalars=None, images=None, texts=None, is_training=None, data_iter_step=1, total_steps=1, should_log_data=None):
		"""
		Args:
			scalars: list of tuples (variable name, value) or
					(variable name, dict{sub_variable_name:sub_variable_value})
			images: tuple (variable name, number of examples, tensor of size BHWC)
			texts: tuple (variable name, list of texts)
			should_log_data: (None|True), use True to force log
		"""
		if self.log_writer is not None:
			if should_log_data is None:
				# compute condition for logging
				should_log_data = (data_iter_step==total_steps-1) or data_iter_step%self.args.log_step==0
				should_log_data = should_log_data and is_training
				should_log_data = should_log_data or (not is_training and data_iter_step==total_steps-1)
			if self.log_writer and should_log_data: 
				# use epoch_1000x as the x-axis in tensorboard to calibrate different
				# curves when batch size changes
				epoch_1000x = int((data_iter_step / total_steps + epoch) * 1000)
				# log scalar values
				for log_name, log_values in scalars:
					if type(log_values) is dict:
						for k, v in log_values.items():
							self.log_writer.add_scalar(f'{sstr}/{log_name}_{k}', v.item() if type(v) is torch.Tensor else v, epoch_1000x)
					else: # scalar value
						self.log_writer.add_scalar(f'{sstr}/{log_name}', log_values.item() if type(log_values) is torch.Tensor else log_values, epoch_1000x)
				# log images
				if images is not None:
					log_name, nb_examples, images = images
					x = len(images) / nb_examples # number of pictures per example
					images = torch.tensor(np.array(images)).permute(0, 3, 1, 2) # convert to torch tensor, and set to BCHW
					grid = make_grid(images, nrow=x, normalize=False) # nrow is actually the number of columns
					self.log_writer.add_image(log_name, grid, epoch_1000x)
				# log texts
				if texts is not None:
					log_name, texts = texts
					for i, t in enumerate(texts):
						self.log_writer.add_text(f'{sstr}/{log_name}_{i}', t, epoch_1000x)


	def add_data_to_metric_logger(self, metric_logger, sstr, scalars):
		"""
		scalars: list of tuples (variable name, value) or (variable name, dict{sub_variable_name:sub_variable_value})
		"""
		d = {}
		for log_name, log_values in scalars:
			if type(log_values) is dict:
				d.update({f'{sstr}_{log_name}_{k}': v.item() if type(v) is torch.Tensor else v for k,v in log_values.items()})
			else: # scalar value
				d[f'{sstr}_{log_name}'] = log_values.item() if type(log_values) is torch.Tensor else log_values
		metric_logger.update( **d )


	def __call__(self):

		# safety checks
		self.check_training_status()
		
		# initializations
		self.init_model()
		self.init_optimizer()
		self.init_lr_scheduler()
		self.init_dataloaders() # may require information about processing pipeline, which depend on the model definition
		self.init_other_training_elements() # needs to be last (depends on previous things)
		
		# load previous ckpt & log parameter names
		self.start_or_resume_training()
		self.print_trainable_params()

		# init tensorboard
		self.log_writer = None # SummaryWriter(log_dir=self.args.output_dir)

		# start training process
		start_time = time.time()
		for epoch in range(self.start_epoch, self.args.epochs): 

			# training
			train_stats = self.training_epoch(epoch)

			# validation
			val_stats = self.validation_epoch(epoch)
			
			# save data
			# -- (ckpt)
			save_best_model = self.has_model_improved(val_stats)
			self.save_checkpoint(save_best_model, epoch)
			# -- (tensorboard)
			if self.log_writer is not None: self.log_writer.flush()
			# -- (regular logs)
			log_stats = {'epoch': epoch, 'time':time.time(), 'lr': self.optimizer.param_groups[0]["lr"]}
			log_stats.update(**train_stats)
			log_stats.update(**val_stats)
			with open(os.path.join(self.args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
				f.write(json.dumps(log_stats) + "\n")

		total_time = time.time() - start_time
		total_time_str = str(datetime.timedelta(seconds=int(total_time)))
		print('Training time {}'.format(total_time_str))