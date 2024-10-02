##############################################################
## poseembroider                                            ##
## Copyright (c) 2024                                       ##
## Institut de Robotica i Informatica Industrial, CSIC-UPC  ##
## and Naver Corporation                                    ##
## Licensed under the CC BY-NC-SA 4.0 license.              ##
## See project root for license details.                    ##
##############################################################

from typing import Union
import torch
import torch.nn as nn
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
if torch.__version__[0] == 2:
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)

from text2pose.encoders.tokenizers import get_text_encoder_or_decoder_module_name
from text2pose.encoders.text_decoders import TransformerTextDecoder, ModalityInputAdapter
from text2pose.generative_modifier.model_generative_modifier import ComparisonModule 

import poseembroider.config as config
from poseembroider.downstream.representation_wrapper import PairRepresentationWrapper


## TEXT GENERATOR
# Input: representation for pose A + representation for pose B
# Output: text
################################################################################

class InstructionGenerator(nn.Module):

    def __init__(self,
                # -- about the input representation
                encoder_latentD=512,
                num_body_joints=config.NB_INPUT_JOINTS,
                path_to_pretrained_representation_model="",
                cached_embeddings_file:Union[bool, dict]=False,
                # -- about the comparison module
                comparison_latentD=512,
                comparison_module_mode="tirg",
                # -- about the text generator
                text_decoder_name="transformer_distilbertUncased",
                transformer_mode="crossattention",
                decoder_latentD=512,
                decoder_nhead=8,
                decoder_nlayers=4,
                ):
        super(InstructionGenerator, self).__init__()

        # Define the data encoder
        self.representation_wrapper = PairRepresentationWrapper(
              num_body_joints=num_body_joints,
              latentD=encoder_latentD,
              path_to_pretrained=path_to_pretrained_representation_model,
              cached_embeddings_file=cached_embeddings_file,
        )

        # Define fusing module
        self.comparison_module = ComparisonModule(inlatentD=encoder_latentD,
                                                  outlatentD=comparison_latentD,
                                                  mode=comparison_module_mode)

        # Define modality input adaptor
        self.modality_input_adapter = ModalityInputAdapter(inlatentD=comparison_latentD,
                                                          outlatentD=decoder_latentD)
        
        # Define text decoder
        self.text_decoder_name = text_decoder_name
        self.transformer_mode = transformer_mode
        module_ref = get_text_encoder_or_decoder_module_name(text_decoder_name)
        if module_ref == "transformer":
            self.text_decoder = TransformerTextDecoder(self.text_decoder_name,
                                                        nhead=decoder_nhead,
                                                        nlayers=decoder_nlayers,
                                                        decoder_latentD=decoder_latentD,
                                                        transformer_mode=transformer_mode)
        else:
            raise NotImplementedError


    def fuse_input_features(self, input_features):
        z = self.comparison_module(input_features["data_A"], input_features["data_B"])
        z = self.modality_input_adapter(z)
        return z
    

    def decode_text(self, z, texts_tokens, texts_lengths, train=False):
        return self.text_decoder(z, texts_tokens, texts_lengths, train=train)


    def forward(self, item, representation_model_input_types=["poses_A", "poses_B"]):
        input_features = self.representation_wrapper(item, representation_model_input_types)
        input_features = {k:v.to(item['texts_tokens'].device) for k,v in input_features.items()} # set on device
        z = self.fuse_input_features(input_features)
        decoded = self.decode_text(z, item["texts_tokens"], item["texts_lengths"], train=True)
        return dict(z=z, **decoded)


    def get_device_of_input(self, input_dict):
        for k,v in input_dict.items():
            if type(v) == torch.Tensor:
                return v.device
            

    def generate_text(self, item, representation_model_input_types=["poses_A", "poses_B"]):
        input_features = self.representation_wrapper(item, representation_model_input_types)
        to_device = self.get_device_of_input(item)
        input_features = {k:v.to(to_device) for k,v in input_features.items()} # set on device
        z = self.fuse_input_features(input_features)
        decoded_texts, likelihood_scores = self.text_decoder.generate_greedy(z)
        return decoded_texts, likelihood_scores