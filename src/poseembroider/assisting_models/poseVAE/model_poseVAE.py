##############################################################
## poseembroider                                            ##
## Copyright (c) 2024                                       ##
## Institut de Robotica i Informatica Industrial, CSIC-UPC  ##
## and Naver Corporation                                    ##
## Licensed under the CC BY-NC-SA 4.0 license.              ##
## See project root for license details.                    ##
##############################################################

import torch
import torch.nn as nn
import numpy as np

from text2pose.encoders.pose_encoder_decoder import PoseDecoder, PoseEncoder

import poseembroider.config as config

class PoseVAE(nn.Module):

    def __init__(self, num_neurons=512, latentD=32, num_body_joints=config.NB_INPUT_JOINTS):
        super(PoseVAE, self).__init__()

        self.latentD = latentD

        # Define pose auto-encoder
        self.pose_encoder = PoseEncoder(num_neurons=num_neurons, latentD=latentD, num_body_joints=num_body_joints, role="generative")
        self.pose_decoder = PoseDecoder(num_neurons=num_neurons, latentD=latentD, num_body_joints=num_body_joints)
        
        # Define learned loss parameters
        self.decsigma_v2v = nn.Parameter( torch.zeros(1) ) # logsigma
        self.decsigma_jts = nn.Parameter( torch.zeros(1) ) # logsigma
        self.decsigma_rot = nn.Parameter( torch.zeros(1) ) # logsigma


    # FORWARD METHODS ----------------------------------------------------------

    def encode_pose(self, pose_body):
        return self.pose_encoder(pose_body)

    def decode_pose(self, z):
        return self.pose_decoder(z)

    def forward(self, poses):
        q_z = self.encode_pose(poses)
        q_z_sample = q_z.rsample()
        ret = {f"{k}_pose":v for k,v in self.decode_pose(q_z_sample).items()}
        ret.update({'q_z': q_z})
        return ret


    # SAMPLE METHODS -----------------------------------------------------------

    def sample_nposes(self, n=1, **kwargs):
        device = self.decsigma_v2v.device
        # sample pose directly from latent space
        z = torch.tensor(np.random.normal(0., 1., size=(n, self.latentD)), dtype=torch.float32, device=device)
        decode_results = self.decode_pose(z)
        return {k: v.view(int(v.shape[0]/n), n, *v.shape[1:]) for k,v in decode_results.items()}

    def sample_meanposes(self):
        device = self.decsigma_v2v.device
        z = torch.zeros(self.latentD, dtype=torch.float32, device=device)
        return self.decode_pose(z.view(1,-1))