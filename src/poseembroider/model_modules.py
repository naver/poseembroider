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
import torch.nn as nn
import roma

from text2pose.encoders.pose_encoder_decoder import PoseEncoder as PoseEncoder_posevae
from text2pose.encoders.text_encoders import TransformerTextEncoder as TextEncoder_posetext

import poseembroider.config as config
os.environ['TORCH_HOME'] = config.TORCH_CACHE_DIR
import poseembroider.utils as utils


# UTILS
################################################################################

class L2Norm(nn.Module):
	def forward(self, x):
		return x / x.norm(dim=-1, keepdim=True)


class AddModule(nn.Module):

	def __init__(self, axis=0):
		super(AddModule, self).__init__()
		self.axis = axis

	def forward(self, x):
		return x.sum(self.axis)


class ConCatModule(nn.Module):

	def __init__(self):
		super(ConCatModule, self).__init__()

	def forward(self, x):
		x = torch.cat(x, dim=1)
		return x


class MiniMLP(nn.Module):

	def __init__(self, input_dim, hidden_dim=None, output_dim=None, normalize=False, three_layers=False):
		super(MiniMLP, self).__init__()
		
		if hidden_dim == output_dim == None:
			hidden_dim = output_dim = input_dim
		elif hidden_dim == None:
			hidden_dim = input_dim

		layers = [
				nn.Linear(input_dim, hidden_dim),
				nn.ReLU(),
		]
		if three_layers:
			layers += [
				nn.Linear(hidden_dim, hidden_dim),
				nn.ReLU(),
				nn.Dropout(0.1),
			]
		layers += [ nn.Linear(hidden_dim, output_dim) ]

		self.layers = nn.Sequential(*layers)
		self.normalize = normalize

		self.init_weights()


	def init_weights(self):
		for layer in self.layers:
			if isinstance(layer, nn.Linear):
				nn.init.trunc_normal_(layer.weight, std=0.02)
				if layer.bias is not None:
					nn.init.zeros_(layer.bias)


	def forward(self, x):
		x = self.layers(x)
		if self.normalize:
			x = nn.functional.normalize(x, dim=-1)
		return x


def average_pooling(token_embeddings, attention_mask=None):
	# take attention mask into account for correct mean pooling of all token embeddings
	batch_size, nbtokens, embed_dim = token_embeddings.shape
	if attention_mask is None: attention_mask = torch.ones(batch_size, nbtokens, device=token_embeddings.device).long()
	input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
	x = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
	return x.view(batch_size, 1, embed_dim)


def get_projection(projection_type, pretrained_output_dim, latentD):
	if projection_type=="layerplus":
		return nn.Sequential(
			nn.Linear(pretrained_output_dim, latentD),
			nn.ReLU(),
			nn.Dropout(0.1),
		)
	elif projection_type=="minimlp":
		return MiniMLP(input_dim=pretrained_output_dim, output_dim=latentD)
	else:
		raise NotImplementedError


# BASE ENCODERS (image)
################################################################################

class ImageEncoder(nn.Module):

	def __init__(self, latentD=512, image_encoder_name='smplerx_vitb32', projection_type="layerplus"):
		super(ImageEncoder, self).__init__()

		self.latentD = latentD

		# define pretrained model
		# Be sure to define the corresponding image processing pipeline in 
		# 		* datasets/base_dataset.py, in `padd_and_resize_keeping_aspect_ratio`
		# 		* augmentations.py, in `get_image_transformation`
		# 		* utils.py, in `get_img_processing_scheme`
		if image_encoder_name == "smplerx_vitb32":

			# define config
			backbone_config = dict(
				img_size=(256, 192),
				patch_size=16,
				embed_dim=768,
				depth=12,
				num_heads=12,
				ratio=1,
				use_checkpoint=False,
				mlp_ratio=4,
				qkv_bias=True,
				drop_path_rate=0.3,
			)
			
			# define backbone
			from poseembroider.smplerx_image_backbone import ViT
			self.pretrained_image_encoder = ViT(**backbone_config)
			
			# load checkpoint
			pretrained_model_path = os.path.join(config.TORCH_CACHE_DIR, "smplerx/smpler_x_b32.pth.tar")
			ckpt = torch.load(pretrained_model_path)
			# extract the backbone weights
			from collections import OrderedDict
			new_state_dict = OrderedDict()
			for k, v in ckpt['network'].items():
				if ('backbone.' in k) or ('encoder.' in k):
					k = k.replace('backbone.', '').replace('encoder.', '')
					if k.startswith('module.'):
						k = k[len("module."):]
					new_state_dict[k] = v
			# load the selected weights
			self.pretrained_image_encoder.load_state_dict(new_state_dict, strict=True)
			print("Initialized image encoder with:", pretrained_model_path)

			self.pretrained_output_dim = self.pretrained_image_encoder.embed_dim
			self.forward = self.forward_smplerx
		
		else:
			raise NotImplementedError
		
		# define projection head for dimension conversion
		self.projection_type = projection_type
		self.image_projection = get_projection(projection_type, self.pretrained_output_dim, self.latentD)


	def set_pretrained_weights_grad(self, value):
		if type(value) is bool:
			for param in self.pretrained_image_encoder.parameters():
				param.requires_grad = value
		else:
			raise NotImplementedError


	def forward_smplerx(self, images, avgpooling_of_tokens=True):
		# NOTE: the code line below yields two outputs: the image features and
		# the task token features (used for regressing pose/shape/camera
		# parameters in the original SMPLer-X model). We are interested in the
		# first output only.
		img_tokens_emb, _ = self.pretrained_image_encoder.forward_features(x=images) # shape (batch_size, pretrained_output_dim, H, W)
		img_tokens_emb = img_tokens_emb.flatten(2,3).permute(0, 2, 1) # shape (batch_size, num_tokens, pretrained_output_dim)
		img_tokens_emb = self.image_projection(img_tokens_emb) # shape (batch_size, num_tokens, latentD)
		# NOTE: here, the average pooling is performed *after* going through the
		# projection MLP; the advantage, compared to doing it *before*, is that
		# it gives a chance to the model to select the most important image
		# features so as to aggregate them better at pooling time. Otherwise, in
		# the "before" setting, useful and useless features are pooled equally.
		if avgpooling_of_tokens:
			img_tokens_emb = average_pooling(img_tokens_emb) # (batch_size, 1, latentD)
		return img_tokens_emb


# BASE ENCODERS (text)
################################################################################

class TextEncoder(nn.Module):

	def __init__(self, latentD=512, text_encoder_name='posetext', projection_type="layerplus"):
		super(TextEncoder, self).__init__()

		self.latentD = latentD

		# define pretrained model
		if text_encoder_name == 'posetext':
			# load checkpoint
			pretrained_model_path = utils.read_json(config.PRETRAINED_MODEL_DICT)["posetext_model_bedlamscript"]			
			ckpt = torch.load(pretrained_model_path, 'cpu')
			assert ckpt['args'].text_encoder_name == "distilbertUncased", "Be sure to propagate possible implications of a different text tokenizer & encoder."
			# initialize encoder
			self.pretrained_text_encoder = TextEncoder_posetext(
												text_encoder_name=ckpt['args'].text_encoder_name,
												latentD=ckpt['args'].latentD,
												topping="avgp",
												role=None)
			# extract relevant pretrained weights
			self.pretrained_text_encoder.load_state_dict({k[len('text_encoder.'):]: v for k,v in ckpt['model'].items() if k.startswith('text_encoder.')}) # NOTE: there is more to the checkpoint than just that text encoder
			print("Initialized text encoder with:", pretrained_model_path)
			
			self.pretrained_output_dim = ckpt['args'].latentD
			self.forward = self.forward_posetext

		else:
			raise NotImplementedError
		
		# define projection head for dimension conversion
		self.projection_type = projection_type
		self.text_projection = get_projection(projection_type, self.pretrained_output_dim, self.latentD)


	def set_pretrained_weights_grad(self, value):
		if type(value) is bool:
			for name, param in self.pretrained_text_encoder.named_parameters():
				if name.startswith("pretrained_text_encoder.") or name.startswith("pretrained_encoder."):
					# freeze distilbert weights
					param.requires_grad = False
				else:
					param.requires_grad = value
		else:
			raise NotImplementedError


	def forward_posetext(self, tokens, token_lengths, avgpooling_of_tokens=True):
		# NOTE: the average pooling is performed by the pretrained model,
		# following the scheme with which it was trained.
		txt_tokens_emb = self.pretrained_text_encoder(tokens, token_lengths) # (batch_size, latentD)
		txt_tokens_emb = self.text_projection(txt_tokens_emb)
		txt_tokens_emb = txt_tokens_emb.view(len(tokens), 1, -1) # shape (batch_size, 1, latentD)
		return txt_tokens_emb


# BASE ENCODERS (pose)
################################################################################

class PoseEncoder(nn.Module):

	def __init__(self, latentD=512, num_body_joints=config.NB_INPUT_JOINTS, pose_encoder_name='posevae', projection_type='layerplus'):
		super(PoseEncoder, self).__init__()

		self.latentD = latentD
		self.num_body_joints = num_body_joints

		# define pretrained model
		if pose_encoder_name == "posevae":
			# load checkpoint
			pretrained_model_path = utils.read_json(config.PRETRAINED_MODEL_DICT)["posevae_model_bedlamscript"]
			ckpt = torch.load(pretrained_model_path, 'cpu')
			assert num_body_joints == ckpt['args'].num_body_joints, f"Mismatch between the desired number of joints ({self.num_body_joints}) and the number of joints in the input of the pretrained pose model ({ckpt['args'].num_body_joints})."
			# initialize encoder
			self.pretrained_pose_encoder = PoseEncoder_posevae(
												latentD=ckpt['args'].latentD,
												num_body_joints=ckpt['args'].num_body_joints,
												role="no_output_layer") # omit the NormalDistDecoder
			# extract relevant pretrained weights
			# (exclude the layers producing the distribution parameters)
			self.pretrained_pose_encoder.load_state_dict({k[len('pose_encoder.'):]: v for k,v in ckpt['model'].items() if k.startswith('pose_encoder.') and 'encoder.8' not in k}) # NOTE: there is more to the checkpoint than just that pose encoder
			print("Initialized pose encoder with:", pretrained_model_path)

			self.pretrained_output_dim = self.pretrained_pose_encoder.encoder[7].weight.shape[1]
			
		else:
			raise NotImplementedError
		
		# define projection head for dimension conversion
		self.projection_type = projection_type
		self.pose_projection = get_projection(projection_type, self.pretrained_output_dim, self.latentD)


	def set_pretrained_weights_grad(self, value):
		if type(value) is bool:
			for param in self.pretrained_pose_encoder.parameters():
				param.requires_grad = value
		else:
			raise NotImplementedError


	def forward(self, poses, avgpooling_of_tokens=True):
		# NOTE: average pooling is not performed because there is only one token
		# per input entry
		ft = self.pretrained_pose_encoder(poses)
		ft = self.pose_projection(ft)
		ft = ft.view(len(poses), 1, -1) # shape (batch_size, number of token, feature_dim)
		return ft


# REGRESSOR MODULES
################################################################################

class IterativeRegressionHead(nn.Module):
	def __init__(self,
				 init_value, # defines the output shape
				 input_dim=512,
				 hidden_dim=1024,
				 **kwargs):
		"""
		Args:
		    init_value: "mean"/standard value for the element to predict. The
				regressor here will learn residuals with respect to that value.
		        It should have the same shape as the expected output (without
		        the batch dimension). 
		"""
		super(IterativeRegressionHead, self).__init__()

		# deduce output info from the init value
		self.output_shape = list(init_value.shape)
		init_value = init_value.flatten()
		output_dim = len(init_value)

		# define model 
		self.regressor_residual = nn.Sequential(
			ConCatModule(), # take the normal input + the latest estimation
			nn.Linear(input_dim + output_dim, hidden_dim),
			nn.Dropout(),
			nn.LeakyReLU(),
			nn.Linear(hidden_dim, hidden_dim),
			nn.Dropout(),
			nn.LeakyReLU(),
			nn.Linear(hidden_dim, output_dim)
		)

		nn.init.xavier_uniform_(self.regressor_residual[-1].weight, gain=0.01)

		self.init_value = nn.Parameter(init_value) # (trainable)

	def forward(self, x, init_value=None, n_iter=3):
		batch_size = x.shape[0]
		if init_value is None:
			init_value = self.init_value.expand(batch_size, -1).clone()

		out = init_value
		for i in range(n_iter):
			out = self.regressor_residual([x, out]) + out
			
		out = out.reshape([batch_size] + self.output_shape)
		return out
	

class PosePredictionHead(IterativeRegressionHead):

	def __init__(self, input_dim=512, num_body_joints=config.NB_INPUT_JOINTS):
		"""
		This module outputs rotation matrices of shape (batch_size, N, 3, 3)
		"""

		# -- Define initialization value
		if os.path.isfile(config.MEAN_SMPLX_POSE_FILE):
			# NOTE: the mean pose below has the same orientation as the poses
			# handled by BEDLAM-Script/-Fix datasets (ie. the global orientation is
			# adapted to the shared framework, which uses a -pi/2 basis rotation
			# for the rendering pipeline) 
			mean_pose = utils.read_pickle(config.MEAN_SMPLX_POSE_FILE)["mean_pose_smplx"] # shape (n_joints, 3)
			mean_pose = mean_pose[:num_body_joints] # (joint rotations in axis angle representation)
			# convert to 6D continuous representation
			# (ie. take the first two columns from the 3x3 rotation matrix)
			mean_pose = roma.rotvec_to_rotmat(mean_pose) # shape (n_joints, 3, 3)
			init_value = mean_pose[:, :, [0,1]] # shape (n_joints, 3, 2)
		else:
			print(f"WARNING! File {config.MEAN_SMPLX_POSE_FILE} not found. Cannot load the precomputed initial values for the iterative regression head. Using a zero-based initialization instead!")
			init_value = torch.zeros(num_body_joints, 3, 2)

		IterativeRegressionHead.__init__(self, input_dim=input_dim, init_value=init_value)

	def forward(self, x):
		x = super().forward(x)
		return roma.special_gramschmidt(x) # rotation matrices: (batch_size, n_joints, 3, 3)