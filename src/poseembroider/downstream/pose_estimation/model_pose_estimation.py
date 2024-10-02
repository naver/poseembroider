##############################################################
## poseembroider                                            ##
## Copyright (c) 2024                                       ##
## Institut de Robotica i Informatica Industrial, CSIC-UPC  ##
## and Naver Corporation                                    ##
## Licensed under the CC BY-NC-SA 4.0 license.              ##
## See project root for license details.                    ##
##############################################################

import os
from typing import Union
import roma
import torch
import torch.nn as nn
import numpy as np
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

import poseembroider.config as config
os.environ['TORCH_HOME'] = config.TORCH_CACHE_DIR

import poseembroider.utils as utils
from poseembroider.model_modules import MiniMLP
from poseembroider.model_modules import PosePredictionHead
from poseembroider.downstream.representation_wrapper import RepresentationWrapper


## HPS ESTIMATOR
# data (eg: image, text) --> 3D pose & shape
################################################################################

class HPSEstimator(nn.Module):

    def __init__(self,
                num_body_joints=config.NB_INPUT_JOINTS,
                num_shape_coeffs=config.NB_SHAPE_COEFFS,
                predict_bodyshape=False,
                # -- about the input representation
                encoder_latentD=512,
                path_to_pretrained_representation_model="",
                cached_embeddings_file:Union[bool, dict]=False,
                ):
        super(HPSEstimator, self).__init__()

        # Define the data encoder
        self.representation_wrapper = RepresentationWrapper(
              num_body_joints=num_body_joints,
              latentD=encoder_latentD,
              path_to_pretrained=path_to_pretrained_representation_model,
              cached_embeddings_file=cached_embeddings_file,
        )

        self.num_body_joints = num_body_joints
        self.num_shape_coeffs = num_shape_coeffs

        # 3D pose prediction head
        self.pose_prediction_head = PosePredictionHead(input_dim=encoder_latentD, num_body_joints=num_body_joints)

        # shape prediction head
        self.predict_bodyshape = predict_bodyshape
        if self.predict_bodyshape:
            self.bodyshape_prediction_head = MiniMLP(input_dim=encoder_latentD, output_dim=num_shape_coeffs, normalize=False)


    def forward(self, item, representation_model_input_types=["images"]):

        input_features = self.representation_wrapper(item, representation_model_input_types).to(item['poses'].device)
        pred_rotmat = self.pose_prediction_head(input_features)
        
        if self.predict_bodyshape:
            pred_betas = self.bodyshape_prediction_head(input_features)
        else:
            pred_betas = None

        return pred_rotmat, pred_betas
    

    def forward_loss(self, item, representation_model_input_types=["images"],
                    loss_type_rotation='geodesic', pose_pred_smpl_losses=False,
                    body_model=None, **kwargs):

        input_features = self.representation_wrapper(item, representation_model_input_types).to(item['poses'].device)

        # preparations
        target_rotmat = roma.rotvec_to_rotmat(item["poses"]) # shape (batch_size, nb joints, 3, 3)
        betas_orig = None
        # --- shape
        if self.predict_bodyshape:
            assert item["shapes"] is not None
            betas_orig = item["shapes"]
        # --- target values for SMPL losses
        if pose_pred_smpl_losses:
            bm_orig = body_model(**utils.pose_data_as_dict(item["poses"], code_base="smplx"), betas=betas_orig)
            joint_set = torch.concat([torch.arange(22), torch.arange(25,25+30)])[:self.num_body_joints] # main body joints (including global rotation) & hand joints
        
        # compute loss terms
        loss_dict = {}
            
        # (1) define which losses should be applied
        apply_shape_loss_for_this_input_bundle = False
        # -- learn to predict the body shape only when the input
        # contains the image
        # NOTE: if the text carries information about the body shape, you can
        # also learn to predict the body shape from text input
        if 'images' in representation_model_input_types:
            apply_shape_loss_for_this_input_bundle = True

        # (2) predict & compute losses consequently
        # --- loss on rotations
        pred_rotmat = self.pose_prediction_head(input_features)
        loss_dict['pred_rots'] = rotation_loss(pred_rotmat,
                            target_rotmat,
                            loss_type=loss_type_rotation,
                            reduction_within_element='sum')

        # --- loss on shape parameters
        if self.predict_bodyshape and apply_shape_loss_for_this_input_bundle:
            pred_betas = self.bodyshape_prediction_head(input_features)
            loss_dict['pred_pose_betas'] = lp_loss(pred_betas, betas_orig, p='smoothl1', reduction_within_element='sum')
        else:
            pred_betas = betas_orig.clone()
        # --- loss on joint & vertex positions
        if pose_pred_smpl_losses:
            bm_pred = body_model(**utils.pose_data_as_dict(roma.rotmat_to_rotvec(pred_rotmat), code_base="smplx"), betas=pred_betas)
            loss_dict['pred_pose_jts'] = lp_loss(bm_pred.joints[:,joint_set], bm_orig.joints[:,joint_set], p='smoothl1') # shape (nb_sample)
            loss_dict['pred_pose_v2v'] = lp_loss(bm_pred.vertices, bm_orig.vertices, p='smoothl1') # shape (nb_sample)

        return loss_dict


def rotation_loss(r_hat, r, loss_type='geodesic', reduction_within_element="mean"):
    """
    Args:
        r_hat, r: tensors of shape (batch_size, ..., 3, 3) containing rotation matrices
        loss_type: ('geodesic'|'l1')
        reduction_within_element: whether to sum or average intermediate terms
    
    Returns:
        scalar (mean reduction over the batch)
    """
    # compute element-wise distance
    if loss_type == 'geodesic':
        l = roma.rotmat_geodesic_distance(r_hat, r, clamping=0.99) # shape (batch_size, ...)
    elif loss_type == 'l1':
        l = ((r_hat - r).abs() + np.log(2.0)).sum([-1, -2]) # shape (batch_size, ...)
    else:
        raise NotImplementedError
    # reduce intermediate dimensions (eg. over the list of joints)
    if len(l.shape) == 2:
        if reduction_within_element == "sum":
            l = l.sum(-1)
        elif reduction_within_element == "mean":
            l = l.mean(-1)
    return l.mean(0) # average over the batch


def lp_loss(r_hat, r, p=2, reduction_within_element="mean"):
    """
    Args:
        r_hat, r: shape (batch_size, nb_points, dimension)
                  eg. (batch_size, nb_joints or nb_vertices, 3)
        reduction_within_element: whether to sum or average intermediate terms
        p: ordinal of the loss (1|2|smoothl1); p=2 will make it a L2 loss,
            p=1 a L1 loss, and p='smoothl1' a mix of both
    Output:
        scalar value
    """
    # compute element-wise loss
    if p in [1,2]:
        l = torch.linalg.norm(r_hat-r, dim=-1, ord=p)
    elif p=='smoothl1':
        l = nn.functional.smooth_l1_loss(input=r_hat, target=r, reduction='none').sum(-1) # sum coeffs, just like in a regular L1/L2
    # reduce intermediate dimensions (eg. over the list of joints)
    if len(l.shape) == 2:
        if reduction_within_element == "sum":
            l = l.sum(-1)
        elif reduction_within_element == "mean":
            l = l.mean(-1)
    return l.mean(0) # average over batch