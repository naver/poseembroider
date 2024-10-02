##############################################################
## poseembroider                                            ##
## Copyright (c) 2024                                       ##
## Institut de Robotica i Informatica Industrial, CSIC-UPC  ##
## and Naver Corporation                                    ##
## Licensed under the CC BY-NC-SA 4.0 license.              ##
## See project root for license details.                    ##
##############################################################

# Utilitary functions for the demo applications.
#
# NOTE:
# * Using a decorator on the functions defined here, and importing these
#   functions in another file run with streamlit will work as intended.
# * Unless this file is executed directly, all `st.something` commands here will
#   be automatically disabled by streamlit. Only the `st.something` commands
#   from the main executed file will work.
# ==> check that `st.` is only used for decorators.

import streamlit as st
import torch
import numpy as np
import smplx
import trimesh

import poseembroider.config as config
import poseembroider.utils as utils


### SETTING
################################################################################

DEVICE = 'cpu'

# colors (must be in format RGB)
COLORS = {
	"grey": [0.7, 0.7, 0.7],
	"red": [1.0, 0.4, 0.4],
	"purple": [0.4, 0.4, 1.0],
	"blue": [0.4, 0.8, 1.0],
	"green": [0.67, 0.9, 0.47],
	"dark-red": [0.59, 0.3, 0.3],
	"white": [1., 1., 1.],
}

### FUNCTIONS
################################################################################


@st.cache_resource
def setup_models(model_paths, checkpoint, _load_model_func, device=DEVICE):
    
    # load models
    models = []
    tokenizer_names = []
    for i, mp in enumerate(model_paths):
        if ".pth" not in mp:
            mp = mp + f"/checkpoint_{checkpoint}.pth"
            print(f"Checkpoint not specified (model {i}). Using {checkpoint} checkpoint.")

        m, ten = _load_model_func(mp, device)
        models.append(m)
        tokenizer_names.append(ten)

    return models, tokenizer_names


@st.cache_resource
def setup_body_model(num_betas=config.NB_SHAPE_COEFFS, device=DEVICE):
    body_model = smplx.create(config.SMPLX_BODY_MODEL_PATH,
        'smplx',
        gender='neutral',
        num_betas=num_betas,
        use_pca = False,
        flat_hand_mean = True,
        batch_size=1)
    body_model.eval()
    body_model.to(device)
    return body_model


def convert_img_for_st(image):
    return torch.permute(image, (1,2,0)).numpy()


def pose_to_image(pose_data, body_model, mv, betas=None, color=[0.4, 0.8, 1.0], viewpoint=[], code_base="smplx"):
    """
    mv: MeshViewer instance
    code_base: ('human_body_prior'|'smplx')
    """
    if type(color) is str: color = COLORS[color]
    bm = body_model(**utils.pose_data_as_dict(pose_data, code_base), betas=betas.view(1,-1) if betas is not None else None)
    if code_base == "smplx":    
        vertices = bm.vertices[0].detach().numpy()
        faces = body_model.faces
    else:
        vertices = bm.v[0].detach().numpy()
        faces = body_model.f
    body_mesh = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_colors=np.tile(color+[1.], (vertices.shape[0], 1)))
    body_mesh.apply_transform(trimesh.transformations.rotation_matrix(-np.radians(90), (1, 0, 0))) # base transformation
    if viewpoint:
        body_mesh.apply_transform(trimesh.transformations.rotation_matrix(np.radians(viewpoint[0]), viewpoint[1]))
    mv.set_static_meshes([body_mesh], [np.eye(4)])
    img = mv.render(render_wireframe=False)
    return img
