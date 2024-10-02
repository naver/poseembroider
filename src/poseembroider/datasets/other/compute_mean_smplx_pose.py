##############################################################
## poseembroider                                            ##
## Copyright (c) 2024                                       ##
## Institut de Robotica i Informatica Industrial, CSIC-UPC  ##
## and Naver Corporation                                    ##
## Licensed under the CC BY-NC-SA 4.0 license.              ##
## See project root for license details.                    ##
##############################################################

import os
import argparse
import torch
import roma
from body_visualizer.mesh.mesh_viewer import MeshViewer

import poseembroider.config as config
import poseembroider.demo as demo
import poseembroider.utils as utils


parser = argparse.ArgumentParser()
parser.add_argument('--display', action="store_true", help='Display the computed mean pose.')
args = parser.parse_args()

# define data location
bedlam_pose_filepath = os.path.join(config.DATA_CACHE_DIR, "bedlamscript_version_overlap30_res224_j16_sf11_split_train_poses_reduced_50000.pkl")

# load pose data
all_poses = utils.read_pickle(bedlam_pose_filepath)['poses'] # dict
all_poses = torch.stack(list(all_poses.values())) # shape (N, n_joints, 3)

# compute average (following: https://naver.github.io/roma/#weighted-rotation-averaging)
x = roma.rotvec_to_rotmat(all_poses) # shape (N, n_joints, 3, 3)
x = x.sum(0) # shape (n_joints, 3, 3)
x = roma.special_procrustes(x) # average, shape (n_joints, 3, 3)
x = roma.rotmat_to_rotvec(x) # shape (n_joints, 3)

mean_pose_bedlam_x = x

# save mean pose
utils.write_pickle({
    "mean_pose_smplx":mean_pose_bedlam_x,
    "info": "Computed over 50k poses extracted from the training set of BEDLAM " + \
            "(selected by farther sampling). Remember to check that the " + \
            "global orientation is as desired (depending on your setting, " + \
            "you may want to apply some 90° rotations around the x-axis of " + \
            "the global orientation.)"
    },
    config.MEAN_SMPLX_POSE_FILE,
    tell=True)


# Optionally display the average pose
if args.display:
    import streamlit as st

    # viewer
    imw, imh = 800, 800
    mv = MeshViewer(width=imw, height=imh, use_offscreen=True)
    RENDERING_COLOR=[0.4, 0.8, 1.0]
    
    # display
    body_model_smplx = demo.setup_body_model()
    img_smplx = demo.pose_to_image(mean_pose_bedlam_x, body_model_smplx, mv, color=[0.4, 0.8, 1.0], code_base="smplx")
    st.image(img_smplx, caption="SMPL-X average pose (BEDLAM-Script 50k)")