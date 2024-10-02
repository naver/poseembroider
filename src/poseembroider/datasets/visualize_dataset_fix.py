##############################################################
## poseembroider                                            ##
## Copyright (c) 2024                                       ##
## Institut de Robotica i Informatica Industrial, CSIC-UPC  ##
## and Naver Corporation                                    ##
## Licensed under the CC BY-NC-SA 4.0 license.              ##
## See project root for license details.                    ##
##############################################################

import streamlit as st
from body_visualizer.mesh.mesh_viewer import MeshViewer

import poseembroider.demo as demo
from poseembroider.datasets.bedlam_fix import BEDLAMFix
from poseembroider.datasets.posefix import PoseFix


### INPUT
################################################################################

# set the testing/validation splits first, for default selection
available_data = {"bedlamfix-overlap30_res224_j16_sf11-in15_out20_t05_sim0709": ["val", "train"],
                  "posefix-H": ["test", "val", "train"]}


### SETUP
################################################################################

# --- viewer
imw, imh = 224, 224 # increase size for better resolution
mv = MeshViewer(width=imw, height=imh, use_offscreen=True)
RENDERING_COLOR=[0.4, 0.8, 1.0]

# --- data
@st.cache_resource
def get_data(dataset_version, split):
    if "bedlam" in dataset_version:
        fs_size = {"val":10000, "train":50000}[split]
        dataset = BEDLAMFix(version=dataset_version,
                            split=split,
                            text_index=0,
                            reduced_set=fs_size,
                            cache=True, load_raw_texts=True,
                            item_format='ip')
    elif "posefix" in dataset_version:
        dataset = PoseFix(version=dataset_version,
                            split=split,
                            text_index=0,
                            cache=True, load_raw_texts=True,
                            item_format='p')
    else:
        raise NotImplementedError

    return dataset


# --- body model
body_model = demo.setup_body_model()


### MAIN APP
################################################################################

# QUERY SELECTION
# ---------------
cols_query = st.columns(2)

# choose dataset
dataset_version = cols_query[0].selectbox("Dataset:", list(available_data.keys()), index=0)
split = cols_query[1].selectbox("Split:", available_data[dataset_version], index=0)
# load dataset
dataset = get_data(dataset_version, split)

# define query input interface
query_type = cols_query[0].selectbox("Query type:", ('Split index', 'ID'))
number = cols_query[1].number_input("Split index or ID:", 0, len(dataset.index_2_id_list)) #, value=500)
st.markdown("""---""")

# get pair_id / item_index
pair_id = number if query_type == 'ID' else dataset.index_2_id_list[number]
item_index = dataset.index_2_id_list.index(number) if query_type == 'ID' else number

# get query data
item = dataset.__getitem__(item_index)
raw_text = dataset.get_raw_text(pair_id)

# QUERY DISPLAY
# -------------

st.write(f"**Data ID:** {pair_id} (index in split: {item_index})")

cols = st.columns(2)

def display_pose_data(data_key, col_ind):
    # image
    if "i" in dataset.item_format:
        cols[col_ind].image(demo.convert_img_for_st(item[f'images_{data_key}']), caption=f"Cropped image {data_key}")
    # pose
    if "p" in dataset.item_format:
        pose_img = demo.pose_to_image(item[f'poses_{data_key}'], body_model, mv, code_base="smplx")
        cols[col_ind].image(pose_img, caption=f"Pose {data_key}")

display_pose_data("A", col_ind=0)
display_pose_data("B", col_ind=1)

st.write("**Annotated text:**")
st.write(f"_{raw_text}_")