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
from functools import partial
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

import poseembroider.config as config
os.environ['TORCH_HOME'] = config.TORCH_CACHE_DIR

from poseembroider.model_modules import ImageEncoder, PoseEncoder, TextEncoder
from poseembroider.model_modules import AddModule, MiniMLP, L2Norm
from poseembroider.loss import symBBC, BBC # used in `eval()` functions



################################################################################
# BASE MODEL
################################################################################

def get_external_encoder_projection(projection_type, pretrained_output_dim, latentD):
	if projection_type=="none":
		return nn.Sequential()
	elif projection_type=="minimlp":
		return MiniMLP(input_dim=pretrained_output_dim, output_dim=latentD)
	elif projection_type=="mediummlpxs":
		return MiniMLP(input_dim=pretrained_output_dim, output_dim=latentD, three_layers=True)
	else:
		raise NotImplementedError


class BaseModel(nn.Module):

	def __init__(self,
				latentD=512,
				l2normalize=False,
				num_body_joints=config.NB_INPUT_JOINTS,
				text_encoder_name='posetext',
				image_encoder_name='smplerx_vitb32',
				pose_encoder_name='posevae',
				encoder_projection_type="layerplus", # part of the encoder architecture
				external_encoder_projection_type="none", # part of the BaseModel architecture, but related to the encoder
				avgpool_input_tokens=True
				):
			super(BaseModel, self).__init__()

			# Save characteristics
			# input/output characteristics
			self.latentD = latentD
			self.num_body_joints = num_body_joints
			# architecture characteristics
			self.pose_encoder_name = pose_encoder_name
			self.text_encoder_name = text_encoder_name
			self.image_encoder_name = image_encoder_name
			self.encoder_projection_type = encoder_projection_type
			self.external_encoder_projection_type = external_encoder_projection_type
			self.l2normalize = l2normalize
			self.avgpool_input_tokens = avgpool_input_tokens
			assert avgpool_input_tokens, \
					"Input modalities could be represented by several tokens"+\
					"(ie. if avgpool_input_tokens=False), but then one needs"+\
					"to define a modality-specific reference embedding"+\
					"(modify `get_modality_global_features` accordingly)."+\
					"Note as well that the PoseEmbroidr with an MLP core"+\
					"requires all modalities to be represented by single vectors."

			# Define modality-specific encoders, with an inner projection layer
			self.pose_encoder = PoseEncoder(pose_encoder_name=self.pose_encoder_name, latentD=self.latentD, num_body_joints=num_body_joints, projection_type=encoder_projection_type)
			self.text_encoder = TextEncoder(text_encoder_name=self.text_encoder_name, latentD=self.latentD, projection_type=encoder_projection_type)
			self.image_encoder = ImageEncoder(image_encoder_name=self.image_encoder_name, latentD=self.latentD, projection_type=encoder_projection_type)
			# freeze pretrained weights (projection layer is trainable!)
			self.pose_encoder.set_pretrained_weights_grad(False)
			self.text_encoder.set_pretrained_weights_grad(False)
			self.image_encoder.set_pretrained_weights_grad(False)

			# Define "external" encoder projections (or inner BaseModel projections)
			self.pose_external_encoder_proj = get_external_encoder_projection(external_encoder_projection_type, latentD, latentD)
			self.text_external_encoder_proj = get_external_encoder_projection(external_encoder_projection_type, latentD, latentD)
			self.image_external_encoder_proj = get_external_encoder_projection(external_encoder_projection_type, latentD, latentD)


	def get_modality_features(self, images=None, poses=None, texts_tokens=None, texts_lengths=None, **kwargs):

		# initialize default values
		image_tokens_emb = None
		pose_tokens_emb = None
		text_tokens_emb = None

		# Encode all available input elements, using pretrained frozen encoders,
		# followed by inner trainable projections and external trainable
		# projections
		# output shape: (batch size, nb tokens, latentD)
		# NOTE: there is always max 1 token because of the average pooling
		if images is not None:
			image_tokens_emb = self.image_encoder(images, avgpooling_of_tokens=self.avgpool_input_tokens)
			image_tokens_emb = self.image_external_encoder_proj(image_tokens_emb)
			if self.l2normalize:
				image_tokens_emb = nn.functional.normalize(image_tokens_emb, dim=-1)
		if poses is not None:
			pose_tokens_emb = self.pose_encoder(poses, avgpooling_of_tokens=self.avgpool_input_tokens)
			pose_tokens_emb = self.pose_external_encoder_proj(pose_tokens_emb)
			if self.l2normalize:
				pose_tokens_emb = nn.functional.normalize(pose_tokens_emb, dim=-1)
		if texts_tokens is not None:
			text_tokens_emb = self.text_encoder(texts_tokens, texts_lengths, avgpooling_of_tokens=self.avgpool_input_tokens)
			text_tokens_emb = self.text_external_encoder_proj(text_tokens_emb)
			if self.l2normalize:
				text_tokens_emb = nn.functional.normalize(text_tokens_emb, dim=-1)

		return dict(
					image_tokens_emb=image_tokens_emb, # (may be None)
					pose_tokens_emb=pose_tokens_emb, # (may be None)
					text_tokens_emb=text_tokens_emb # (may be None)
					)


	def get_modality_global_features(self, modality_features=None,
								  images=None, poses=None, texts_tokens=None, texts_lengths=None, **kwargs):
		
		if modality_features is None:
			modality_features = self.get_modality_features(images, poses, texts_tokens, texts_lengths)

		# Prepare reference, modality-specific embeddings
		# If the input is composed of several tokens, the reference embedding
		# can be for instance the CLS token, or the average pooling of the
		# tokens. If the input is represented by a single token, that token is
		# the reference token itself.
		# This latter case is the current one, so we simply extract the provided
		# unique token to be the reference, "global" modality-specific
		# embedding.
		# shape: (batch_size, self.latentD)
		image_emb = modality_features['image_tokens_emb'][:,0] if modality_features['image_tokens_emb'] is not None else None
		pose_emb = modality_features['pose_tokens_emb'][:,0] if modality_features['pose_tokens_emb'] is not None else None
		text_emb = modality_features['text_tokens_emb'][:,0] if modality_features['text_tokens_emb'] is not None else None

		return dict(
					image_emb=image_emb, # (may be None)
					pose_emb=pose_emb, # (may be None)
					text_emb=text_emb # (may be None)
					)


	def get_batch_size_from_ft_dict(self, ft_dict):
		for v in ft_dict.values():
			if v is not None:
				return v.shape[0]


	def get_query_features(self, images=None, poses=None, texts_tokens=None, texts_lengths=None,
								query_modalities=["image", "pose", "text"], **kwargs):
		# implemented in child classes
		raise NotImplementedError



################################################################################
# PoseEmbroider
################################################################################

class PoseEmbroider(BaseModel):
	
	def __init__(self,
			  latentD=512,
			  l2normalize=False,
			  num_body_joints=config.NB_INPUT_JOINTS,
			  text_encoder_name='posetext',
			  image_encoder_name='smplerx_vitb32',
			  pose_encoder_name='posevae',
			  encoder_projection_type='layerplus',
			  external_encoder_projection_type="none",
			  avgpool_input_tokens=True,
			  no_projection_heads=False,
			  embroider_core_type='transformer',
			  nlayers=4, nhead=4, dim_feedforward=1024, activation="gelu", dropout=0.1):
		super(PoseEmbroider, self).__init__(latentD=latentD,
									    l2normalize=l2normalize,
										num_body_joints=num_body_joints,
										avgpool_input_tokens=avgpool_input_tokens,
								  		text_encoder_name=text_encoder_name,
										image_encoder_name=image_encoder_name,
										pose_encoder_name=pose_encoder_name,
										encoder_projection_type=encoder_projection_type,
										external_encoder_projection_type=external_encoder_projection_type)

		# Characteristics
		self.no_projection_heads = no_projection_heads

		# Define learnable tokens
		self.intermodality_token = nn.Parameter(torch.zeros(1, 1, self.latentD))

		# Define modality encoding (akin to learnable positional encodings)
		self.modality_encoding_image = nn.Parameter(torch.zeros(1, 1, self.latentD))
		self.modality_encoding_pose = nn.Parameter(torch.zeros(1, 1, self.latentD))
		self.modality_encoding_text = nn.Parameter(torch.zeros(1, 1, self.latentD))

		# Define the core architecture
		self.embroider_core_type = embroider_core_type
		if self.embroider_core_type == "transformer":
			embroider_core_layers = nn.TransformerEncoderLayer(d_model=self.latentD//2,
										nhead=nhead,
										dim_feedforward=dim_feedforward//2,
										dropout=dropout,
										activation=activation,
										batch_first=True)
			self.embroider_core = nn.Sequential(
				nn.Linear(self.latentD, self.latentD//2),
				embroider_core_layers,
				nn.Linear(self.latentD//2, self.latentD)
			)
			self.norm = partial(nn.LayerNorm, eps=1e-6)(self.latentD)
			self.forward_in_embroider_core = self.forward_in_embroider_core_transformer
		elif self.embroider_core_type == "mlp":
			self.embroider_core = nn.Sequential(
				AddModule(1), # processing inputs of shape (batch_size, nb_tokens, latentD), performing addition of tokens
				MiniMLP(self.latentD, normalize=True)
			)
			self.forward_in_embroider_core = self.forward_in_embroider_core_mlp
		else:
			raise NotImplementedError
		
		# Define projection layers
		if self.no_projection_heads:
			self.modality_projection_image = nn.Sequential(L2Norm()) if self.l2normalize else nn.Sequential()
			self.modality_projection_pose = nn.Sequential(L2Norm()) if self.l2normalize else nn.Sequential()
			self.modality_projection_text = nn.Sequential(L2Norm()) if self.l2normalize else nn.Sequential()
		else:
			self.modality_projection_image = MiniMLP(self.latentD,
													 output_dim=self.latentD,
													 normalize=self.l2normalize)
			self.modality_projection_pose = MiniMLP(self.latentD,
													 output_dim=self.latentD,
													 normalize=self.l2normalize)
			self.modality_projection_text = MiniMLP(self.latentD,
													 output_dim=self.latentD,
													 normalize=self.l2normalize)

		# Define other learnable parameters (eg. temperature)
		self.temperature = torch.nn.Parameter( torch.FloatTensor((10,)) )

		# Initialize weights
		self.init_weights()


	def init_weights(self):
		nn.init.normal_(self.intermodality_token, std=1e-6)


	def forward_in_embroider_core_transformer(self, x):
		"""
		Args:
			x: shape (batch_size, total number of tokens, self.latentD)
				where the total number of tokens comprises the number of tokens
				brought by each input modality, plus the intermodality token 
				(in first position).
		        Note that positional encoding (ie. modality encoding) should
				already have been applied.
		
		Return:
			shape (batch_size, self.latentD), intermodality token
		"""
		x = self.embroider_core(x)
		x = self.norm(x)
		return x[:,0]


	def forward_in_embroider_core_mlp(self, x):
		"""
		Args:
			x: shape (batch_size, total number of tokens, self.latentD)
				where the total number of tokens corresponds to the number of
		        input modalities, plus the intermodality token (in first
		        position).
		
		Return:
			shape (batch_size, self.latentD), intermodality token
		"""
		x = x[:,1:] # ignore the learnable intermodality token
		x = self.embroider_core(x) # the output is considered to be the intermodality token
		return x


	def get_modality_projections(self, intermodality_token):
		"""
		Args:
			intermodality_token: (batch_size, self.latentD)

		Return:
		    dict {k: (batch_size, self.latentD) }, predicted global token for
		    modality k
		"""
		# get modality-specific projections of the intermodality token
		x_ret = {}
		for m in ['image', 'pose', 'text']: # iterate over each modality to predict
			# project in the corresponding modality space
			x_ret[f'predicted_{m}'] = eval(f'self.modality_projection_{m}')(intermodality_token)
		return x_ret

	
	def project_for_contrastive_losses(self, x, x_ref_dict, loss_type, prefix=""):
		"""
		Project the intermodality token in each modality's space, then compute
		the contrastive loss between it and the original modality's global token.

		Args:
			x: intermodality token, shape (batch_size, self.latentD)
			x_ref_dict: dict {k: global token for modality k,
									shape (batch_size, self.latentD)}
			loss_type: loss function to apply
			prefix: prefix for the keys of the different loss terms in the
					returned dictionary.

		Return:
			l_dict: dict {k: contrastive loss between the modality-specific
								projection of the intermodality token
								and the global token for modality k, float}
		"""
		l_dict = {}
		for k in x_ref_dict:
			k_ = k.replace('_emb', '')
			x_proj = eval(f'self.modality_projection_{k_}')(x)
			scores = x_proj.mm(x_ref_dict[k].t())
			l_dict[prefix+k_] = eval(loss_type)(scores * self.temperature)
		return l_dict


	def forward(self, images, poses, texts_tokens, texts_lengths,
					single_partials=False, dual_partials=True, triplet_partial=True,
					loss_type="symBBC",
					**kwargs):
			
		# Get modality feature tokens
		x = self.get_modality_features(images, poses, texts_tokens, texts_lengths) # {"*_tokens_emb": (batch_size, nb_tokens=1, latentD) }
		
		# Get global features for each modality
		x_ref_dict = self.get_modality_global_features(modality_features=x) # {"*_emb": (batch_size, latentD) }

		# Apply modality-specific encoding
		x['image_tokens_emb'] += self.modality_encoding_image
		x['pose_tokens_emb'] += self.modality_encoding_pose
		x['text_tokens_emb'] += self.modality_encoding_text

		# Concat all input elements (build different input bundles) &
		# Forward through the embroider_core &
		# Get the output intermodality token for each input bundle
		# input: (batch_size, nb_tokens, latentD), 
		# - order: intermodality token > image > pose > text (alphabetic)
		# output: (batch size, latentD)
		batch_size = self.get_batch_size_from_ft_dict(x)
		intermodality_token = self.intermodality_token.expand(batch_size, -1, -1)
		x_dict = {}
		if single_partials:
			x_dict.update({
				# only the image
				'only_image_input': self.forward_in_embroider_core(torch.cat((intermodality_token.clone(), x['image_tokens_emb'].clone()), dim=1)),
				# only the pose
				'only_pose_input': self.forward_in_embroider_core(torch.cat((intermodality_token.clone(), x['pose_tokens_emb'].clone()), dim=1)),
				# only the text
				'only_text_input': self.forward_in_embroider_core(torch.cat((intermodality_token.clone(), x['text_tokens_emb'].clone()), dim=1)),
			})
		if dual_partials:
			x_dict.update({
				# pose + text (missing image)
				'missing_image_input': self.forward_in_embroider_core(torch.cat((intermodality_token.clone(), x['pose_tokens_emb'].clone(), x['text_tokens_emb'].clone()), dim=1)),
				# image + text (missing pose)
				'missing_pose_input': self.forward_in_embroider_core(torch.cat((intermodality_token.clone(), x['image_tokens_emb'].clone(), x['text_tokens_emb'].clone()), dim=1)),
				# image + pose (missing text)
				'missing_text_input': self.forward_in_embroider_core(torch.cat((intermodality_token.clone(), x['image_tokens_emb'].clone(), x['pose_tokens_emb'].clone()), dim=1)),
			})
		if triplet_partial:
			# add full input, image + pose + text
			x_dict['full_input'] = self.forward_in_embroider_core(torch.cat((intermodality_token.clone(), x['image_tokens_emb'].clone(), x['pose_tokens_emb'].clone(), x['text_tokens_emb'].clone()), dim=1))

		# Compute the different loss terms
		# between modality-specific projections of the intermodality token and
		# each original modality-specific global token
		loss_dict = {}
		for k in x_dict: # iterate over each input bundle
			l = self.project_for_contrastive_losses(x_dict[k], x_ref_dict, loss_type, prefix=f'{k}_clossw_',)
			loss_dict.update(l)

		return loss_dict
		

	def get_intermodality_tokens_for_all_query_types(self, images=None, poses=None, texts_tokens=None, texts_lengths=None, **kwargs):
		"""
		Get the intermodality token for all possible combinations of (partial)
		input bundles that can be obtained from the given modalities.

		Return:
		    x_dict: dict {k: (batch_size, self.latentD) }, of intermodality
		    		tokens (value) for different bundle inputs (key)
		"""
		
		# Get modality feature tokens
		mf = self.get_modality_features(images, poses, texts_tokens, texts_lengths)

		# Apply modality-specific encoding
		if mf['image_tokens_emb'] is not None: mf['image_tokens_emb'] += self.modality_encoding_image
		if mf['pose_tokens_emb'] is not None: mf['pose_tokens_emb'] += self.modality_encoding_pose
		if mf['text_tokens_emb'] is not None: mf['text_tokens_emb'] += self.modality_encoding_text

		# Concat all input elements (for all different input bundles)
		# - order: intermodality token > image > pose > text (alphabetic)
		batch_size = self.get_batch_size_from_ft_dict(mf)
		intermodality_token = self.intermodality_token.expand(batch_size, -1, -1)
		x_dict = {}
		# image only
		if images is not None: x_dict['images_input'] = torch.cat((intermodality_token.clone(), mf['image_tokens_emb'].clone()), dim=1)
		# pose only
		if poses is not None: x_dict['poses_input'] = torch.cat((intermodality_token.clone(), mf['pose_tokens_emb'].clone()), dim=1)
		# text only
		if texts_tokens is not None: x_dict['texts_input'] = torch.cat((intermodality_token.clone(), mf['text_tokens_emb'].clone()), dim=1)
		# image + pose (missing text)
		if images is not None and poses is not None: x_dict['images_poses_input'] = torch.cat((intermodality_token.clone(), mf['image_tokens_emb'].clone(), mf['pose_tokens_emb'].clone()), dim=1)
		# pose + text (missing image)
		if poses is not None and texts_tokens is not None: x_dict['poses_texts_input'] = torch.cat((intermodality_token.clone(), mf['pose_tokens_emb'].clone(), mf['text_tokens_emb'].clone()), dim=1)
		# image + text (missing pose)
		if images is not None and texts_tokens is not None: x_dict['images_texts_input'] = torch.cat((intermodality_token.clone(), mf['image_tokens_emb'].clone(), mf['text_tokens_emb'].clone()), dim=1)

		# Forward through the embroider_core and get the intermodality token for
		# each input bundle
		for k in x_dict:
			x_dict[k] = self.forward_in_embroider_core(x_dict[k])

		return x_dict


	def get_intermodality_token(self, images=None, poses=None, texts_tokens=None, texts_lengths=None,
								query_modalities=["image", "pose", "text"], **kwargs):
		"""
		Get the intermodality token for the combination of the provided input
		modalities.

		Return:
		    (batch_size, self.latentD) }, intermodality token
		"""

		# Get modality feature tokens
		mf = self.get_modality_features(images, poses, texts_tokens, texts_lengths)

		# Apply modality-specific encoding
		if mf['image_tokens_emb'] is not None: mf['image_tokens_emb'] += self.modality_encoding_image
		if mf['pose_tokens_emb'] is not None: mf['pose_tokens_emb'] += self.modality_encoding_pose
		if mf['text_tokens_emb'] is not None: mf['text_tokens_emb'] += self.modality_encoding_text

		# Concat all input elements
		# - order: intermodality token > image > pose > text (alphabetic)
		batch_size = self.get_batch_size_from_ft_dict(mf)
		intermodality_token = self.intermodality_token.expand(batch_size, -1, -1)
		x_input = [intermodality_token]
		if "image" in query_modalities:
			x_input += [mf['image_tokens_emb']]
		if "pose" in query_modalities:
			x_input += [mf['pose_tokens_emb']]
		if "text" in query_modalities:
			x_input += [mf['text_tokens_emb']]
		x_input = torch.cat(x_input, dim=1)

		# Forward through the embroider_core and get the intermodality token for
		# this input
		x_input = self.forward_in_embroider_core(x_input)

		return x_input
	

	def get_query_features(self, images=None, poses=None, texts_tokens=None, texts_lengths=None,
								query_modalities=["image", "pose", "text"], **kwargs):
		"""
		Get all modality-specific reprojections of the intermodality token
		obtained from the combination of the provided input modalities.

		Return:
		    x_ret: dict {k: (batch_size, self.latentD) }, predicted global token
		    		for modality k (modality-specific reprojection)
			x_input: (batch_size, self.latentD), intermodality token
		"""

		# Get intermodality token for this input
		x_input = self.get_intermodality_token(images, poses, texts_tokens, texts_lengths, query_modalities)

		# Prepare output: modality-specific projections of the intermodality token
		x_ret = self.get_modality_projections(x_input)

		# return:
		# - the predicted projections,
		# - the intermodality token
		return x_ret, x_input



################################################################################
# ALIGNER
################################################################################

class Aligner(BaseModel):

	def __init__(self,
			  latentD=512,
			  l2normalize=False,
			  num_body_joints=config.NB_INPUT_JOINTS,
			  text_encoder_name='posetext',
			  image_encoder_name='smplerx_vitb32',
			  pose_encoder_name='posevae',
			  encoder_projection_type='layerplus',
			  external_encoder_projection_type="mediummlpxs",
			  avgpool_input_tokens=True,
			  ):
		super(Aligner, self).__init__(latentD=latentD,
									    l2normalize=l2normalize,
										num_body_joints=num_body_joints,
										avgpool_input_tokens=avgpool_input_tokens,
								  		text_encoder_name=text_encoder_name,
										image_encoder_name=image_encoder_name,
										pose_encoder_name=pose_encoder_name,
										encoder_projection_type=encoder_projection_type,
										external_encoder_projection_type=external_encoder_projection_type)

		# Define other learnable parameters (eg. temperature)
		self.temperature = torch.nn.Parameter( torch.FloatTensor((10,)) )


	def mix_singles(self, x_list):
		"""
		Args:
			x_list: list of embeddings

		Return:
			L2-normalized average of the input embeddings (intermodality token)
		"""
		q = torch.zeros_like(x_list[0])
		for x in x_list:
			q += x
		q /= len(x_list)
		if self.l2normalize:
			q = nn.functional.normalize(q, dim=-1)
		return q


	def get_contrastive_losses(self, x_ref_dict, loss_type, prefix="",
								single_partials=True,
								dual_partials=False,
								triplet_partial=False):
		"""
		Args:
			x_ref_dict: dict {k: global token for modality k,
									shape (batch_size, self.latentD)}
			loss_type: loss function to apply
			prefix: prefix for the keys of the different loss terms in the
					returned dictionary.

		Return:
		    l_dict: dict {k: contrastive loss between the intermodality token
		    				(average of the input global modality tokens)
		                    and the global token for modality k, float}
		"""
		l_dict = {}
		parse_m = lambda m: m.replace('_emb', '')
		modalities = list(x_ref_dict.keys())
		# iterate over each modality
		for m1i, m1 in enumerate(modalities):
			m1n = parse_m(m1)
			# iterate over each remaining modality
			for m2i, m2 in enumerate(modalities):
				if m1i < m2i:
					m2n = parse_m(m2)
					# get contrastive loss
					# - single-to-single
					if single_partials:
						scores = x_ref_dict[m1].mm(x_ref_dict[m2].t())
						l_dict[f"{prefix}{m1n}_{m2n}"] = eval(loss_type)(scores * self.temperature)
					# - dual-to-single
					if dual_partials:
						# use the last remaining modality as target 
						m3 = [m for m in modalities if (m!=m1 and m!=m2)][0]
						m3n = parse_m(m3)
						# combine input modalities
						q = self.mix_singles([x_ref_dict[m1], x_ref_dict[m2]])
						scores = q.mm(x_ref_dict[m3].t())
						l_dict[f"{prefix}{m1n}-{m2n}_{m3n}"] = eval(loss_type)(scores * self.temperature)
			# - triplet-to-single
			if triplet_partial:
				# combine all modalities
				q = self.mix_singles([x_ref_dict[m] for m in modalities])
				scores = q.mm(x_ref_dict[m1].t())
				l_dict[f"{prefix}full-input_{m1n}"] = eval(loss_type)(scores * self.temperature)

		return l_dict


	def forward(self, images, poses, texts_tokens, texts_lengths,
				single_partials=True, dual_partials=False, triplet_partial=False,
				loss_type="symBBC",
				**kwargs):

		# Get global features for each modality
		# (the Aligner can only process single-token features)
		mgf = self.get_modality_global_features(images=images, poses=poses, texts_tokens=texts_tokens, texts_lengths=texts_lengths)

		# Compute the different loss terms
		# between different intermodality embeddings (obtained by combining
		# different modalities) and the original modality embeddings
		loss_dict = self.get_contrastive_losses(mgf, loss_type,
										  		single_partials=single_partials,
												dual_partials=dual_partials,
												triplet_partial=triplet_partial)
		return loss_dict
		

	def get_query_features(self, images=None, poses=None, texts_tokens=None, texts_lengths=None,
								query_modalities=["image", "pose", "text"], **kwargs):
		"""
		Get the combination of the modality-specific embeddings for the set of
		given query modality.

		Return:
			(batch_size, self.latentD), intermodality embedding 
		"""
		
		# get global features for the given query elements
		mgf = self.get_modality_global_features(images=images, poses=poses, texts_tokens=texts_tokens, texts_lengths=texts_lengths)

		# get mixed feature
		query_features = self.mix_singles([mgf[f'{qm}_emb'] for qm in query_modalities])

		return query_features
	
	
	def get_query_features_from_precomputed_features(self, modality_features,
												  query_modalities=["image", "pose", "text"]):
		"""
		Args:
			modality_features: dict with keys 'image', 'pose', 'text', containing
							the global features for each modality.
			query_modalities: list of the modalities to combine

		Return:
			(batch_size, self.latentD), intermodality embedding
		"""

		# get mixed feature
		query_features = self.mix_singles([modality_features[qm] for qm in query_modalities])

		return query_features