##############################################################
## poseembroider                                            ##
## Copyright (c) 2024                                       ##
## Institut de Robotica i Informatica Industrial, CSIC-UPC  ##
## and Naver Corporation                                    ##
## Licensed under the CC BY-NC-SA 4.0 license.              ##
## See project root for license details.                    ##
##############################################################

import streamlit as st
import argparse
import torch
import numpy as np
from body_visualizer.mesh.mesh_viewer import MeshViewer

import poseembroider.utils as utils
import poseembroider.demo as demo
from poseembroider.datasets.bedlam_fix import BEDLAMFix
from poseembroider.datasets.posefix import PoseFix
from poseembroider.augmentations import DataProcessingModule
from poseembroider.downstream.instruction_generation.evaluate_instruction_generation import load_model


parser = argparse.ArgumentParser(description='Parameters for the demo.')
parser.add_argument('--model_paths', nargs='+', type=str, help='Path to the models to be compared.')
parser.add_argument('--checkpoint', default='best', choices=('best', 'last'), help="Checkpoint to choose if model path is incomplete.")
args = parser.parse_args()


### SETTINGS
################################################################################

COMPUTING_DEVICE = torch.device('cuda:0')

# set the testing/validation splits first, for default selection
available_data = {"bedlamfix-overlap30_res224_j16_sf11-in15_out20_t05_sim0709": ["val", "train"],
                  "posefix-H": ["test", "val", "train"],
                }


### SETUP
################################################################################

# --- models
models, tokenizer_names = demo.setup_models(args.model_paths, args.checkpoint, load_model, device=COMPUTING_DEVICE)
assert len(set([m.representation_wrapper.num_body_joints for m in models])) == 1, "Comparing models using different numbers of joints to represent the pose. Case not handled here. Make the necessary modifications first."
assert len(set([utils.get_img_processing_scheme(m.representation_wrapper.representation_model.image_encoder_name) for m in models])) == 1, "Comparing models with different image processing schemes. Case not handled here. Make the necessary modifications first."
num_body_joints = models[0].representation_wrapper.num_body_joints
img_processing_scheme = utils.get_img_processing_scheme(models[0].representation_wrapper.representation_model.image_encoder_name)
data_processing = DataProcessingModule(phase="eval", nb_joints=None, img_processing_scheme=img_processing_scheme) # we only care about image processing

body_model = demo.setup_body_model()
body_model.eval()
body_model.to('cpu')

# --- data
@st.cache_resource
def get_data(dataset_version, split):
    if "bedlam" in dataset_version:
        fs_size = {"val":10000, "train":50000}[split]
        dataset = BEDLAMFix(version=dataset_version,
                            split=split,
                            reduced_set=fs_size,
                            num_body_joints=num_body_joints,
                            img_processing_scheme=img_processing_scheme,
                            text_index=0,
                            load_raw_texts=True,
                            item_format='ip')
    elif "posefix" in dataset_version:
        dataset = PoseFix(version=dataset_version,
                            split=split,
                            num_body_joints=num_body_joints,
                            text_index=0,
                            load_raw_texts=True,
                            item_format='p')
    else:
        raise NotImplementedError
    return dataset

# --- viewer
imw, imh = 224, 224 # increase size for better resolution
mv = MeshViewer(width=imw, height=imh, use_offscreen=True)
RENDERING_COLOR=[0.4, 0.8, 1.0]

# --- layout
st.markdown("""
            <style>
            .smallfont {
                font-size:10px !important;
            }
            </style>
            """, unsafe_allow_html=True)


### MAIN APP
################################################################################

# QUERY SELECTION
# ---------------
cols_query = st.columns(2)

# choose dataset
dataset_version = cols_query[0].selectbox("Dataset:", list(available_data.keys()), index=1)
split = cols_query[1].selectbox("Split:", available_data[dataset_version], index=0)
# load dataset
dataset = get_data(dataset_version, split)

# define query input interface
query_type = cols_query[0].selectbox("Query type:", ('Split index', 'ID'))
number = cols_query[1].number_input("Split index or ID:", 0, len(dataset.index_2_id_list))
st.markdown("""---""")

# get pair_id / item_index
pair_id = number if query_type == 'ID' else dataset.index_2_id_list[number]
item_index = dataset.index_2_id_list.index(number) if query_type == 'ID' else number

# get query data
item = dataset.__getitem__(item_index)
raw_text = dataset.get_raw_text(pair_id, 0) if 't' in dataset.item_format else ''


# QUERY DISPLAY & INPUT CHOICE
# ----------------------------

st.write(f"**Data ID:** {pair_id} (index in split: {item_index})")

# form initialization
form = st.form("query_form")
form.write("Select what to use as input.")
cols_input = form.columns(2)

representation_model_input_types = []

def display_pose_data_and_set_input_type(data_key, representation_model_input_types):

    # image
    if 'i' in dataset.item_format:
        use_image = cols_input[0].checkbox(f"**Use image {data_key}**")
        if use_image: representation_model_input_types+=[f"images_{data_key}"]
        cols_input[0].image(demo.convert_img_for_st(item[f'images_{data_key}']), caption=f"Image {data_key}")
    else:
        cols_input[0].image(np.ones((imw, imh)), caption="No image available.")

    # pose
    if 'p' in dataset.item_format:
        use_pose = cols_input[1].checkbox(f"**Use pose {data_key}**", value=True)
        if use_pose: representation_model_input_types+=[f"poses_{data_key}"]
        pose_img = demo.pose_to_image(item[f'poses_{data_key}'], body_model, mv, code_base="smplx")
        cols_input[1].image(pose_img, caption=f"Pose {data_key}")
    else:
        cols_input[1].image(np.ones((imw, imh)), caption="No pose available.")

    return representation_model_input_types

representation_model_input_types = display_pose_data_and_set_input_type("A", representation_model_input_types)
representation_model_input_types = display_pose_data_and_set_input_type("B", representation_model_input_types)

# add the form submit button
submit_input = form.form_submit_button("Generate text!")


# DISPLAY REFERENCE TEXT
# ----------------------

st.write("**Annotated text:**")
st.write(f"_{raw_text}_")


# TEXT GENERATION
# ---------------

if submit_input and len(representation_model_input_types):
    
    st.markdown("""---""")
    st.write("**Text generation:**")

    for i, model in enumerate(models):

        with torch.no_grad():
            # prepare input
            input_dict = {k:v.unsqueeze(0) for k,v in item.items() if k not in ['indices', 'texts_tokens', 'texts_lengths', 'data_ids_A', 'data_ids_B', 'pair_ids', 'dataset']}
            input_dict = data_processing(input_dict) # process images
            input_dict = {k:v.to(COMPUTING_DEVICE) for k,v in input_dict.items()}
            
            # generate text
            texts, scores = model.generate_text(item=input_dict, representation_model_input_types=representation_model_input_types) # (1, njoints, 3)

        if len(models) > 1:
            cols = st.columns(2)
            cols[0].markdown(f'<p class="smallfont">{args.model_paths[i]}</p>', unsafe_allow_html=True)
            cols[1].write(texts[0])
            st.markdown("""---""")
        else:
            st.write(texts[0])
            st.markdown("""---""")
            st.write(f"_Results obtained with model: {args.model_paths[0]}_")

elif submit_input and len(representation_model_input_types)==0:
    st.markdown("""---""")
    st.warning("Need to select input data!")