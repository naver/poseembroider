##############################################################
## poseembroider                                            ##
## Copyright (c) 2024                                       ##
## Institut de Robotica i Informatica Industrial, CSIC-UPC  ##
## and Naver Corporation                                    ##
## Licensed under the CC BY-NC-SA 4.0 license.              ##
## See project root for license details.                    ##
##############################################################

import torch
import os

from text2pose.encoders.tokenizers import get_tokenizer_name
from text2pose.retrieval.model_retrieval import PoseText


def load_model(model_path, device):
	
	assert os.path.isfile(model_path), "File {} not found.".format(model_path)
	
	# load checkpoint & model info
	ckpt = torch.load(model_path, 'cpu')
	text_encoder_name=ckpt['args'].text_encoder_name

	# load model
	model = PoseText(text_encoder_name=text_encoder_name,
				  	 transformer_topping=ckpt['args'].transformer_topping,
					 latentD=ckpt['args'].latentD,
					 num_body_joints=ckpt['args'].num_body_joints,
					 ).to(device)
	model.load_state_dict(ckpt['model'])
	model.eval()
	print(f"Loaded model from (epoch {ckpt['epoch']}):", model_path)

	return model, get_tokenizer_name(text_encoder_name)
