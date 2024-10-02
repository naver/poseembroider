##############################################################
## poseembroider                                            ##
## Copyright (c) 2024                                       ##
## Institut de Robotica i Informatica Industrial, CSIC-UPC  ##
## and Naver Corporation                                    ##
## Licensed under the CC BY-NC-SA 4.0 license.              ##
## See project root for license details.                    ##
##############################################################

import streamlit as st
import roma
from body_visualizer.mesh.mesh_viewer import MeshViewer

import poseembroider.demo as demo
from poseembroider.datasets.bedlam_script import BEDLAMScript
from poseembroider.datasets.threedpw import ThreeDPW


### INPUT
################################################################################

# set the testing/validation splits first, for default selection
available_data = {
    "bedlamscript-overlap30_res224_j16_sf11": ["val", "train"],
    "threedpw-sf1": ["test", "val"],
}


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
        dataset = BEDLAMScript(version=dataset_version,
                            split=split,
                            text_index=0,
                            reduced_set=fs_size,
                            cache=True,
                            load_raw_texts=True,
                            item_format='ip')
    elif "threedpw" in dataset_version:
        dataset = ThreeDPW(version=dataset_version,
                           split=split,
                           cache=True,
                           item_format='ip')
    else:
        raise NotImplementedError

    return dataset


### MAIN APP
################################################################################

# QUERY SELECTION
# ---------------
cols_query = st.columns(2)

# choose dataset
dataset_version = cols_query[0].selectbox("Dataset:", list(available_data.keys()), index=0)
split_for_research = cols_query[1].selectbox("Split:", available_data[dataset_version], index=0)
# load dataset
dataset = get_data(dataset_version, split_for_research)
# body model
body_model = demo.setup_body_model(num_betas=dataset.num_shape_coeffs)

# define query input interface
cols_query = st.columns(2)
query_type = cols_query[0].selectbox("Query type:", ('Split index', 'ID'))
number = cols_query[1].number_input("Split index or ID:", 0, len(dataset.index_2_id_list))
st.markdown("""---""")

# get data_id / item_index
data_id = number if query_type == 'ID' else dataset.index_2_id_list[number]
item_index = dataset.index_2_id_list.index(number) if query_type == 'ID' else number

# get query data
item = dataset.__getitem__(item_index)

# QUERY DISPLAY
# -------------

# image
if "i" in dataset.item_format:
    st.image(demo.convert_img_for_st(item[f'images']), caption=f"Cropped image")
    print(dataset.images[data_id])

# pose
if "p" in dataset.item_format:
    pose_img_cols = st.columns(2)
    pose = item['poses'].clone()
    betas = None
    # render the pose in the camera reference frame
    if 'cam_rot' in item:
        use_cam_rot = pose_img_cols[1].checkbox("Apply camera transformation.")
        if use_cam_rot:
            pose[0:1] = roma.rotvec_composition([item['cam_rot'].view(1,3), item['poses'][0:1].clone()])
    # render the body with a particular shape
    # (default (eg. when shape (betas) not available): displaying the body with the default shape (neutral body model))
    betas = None
    if "shapes" in item:
        use_shape = pose_img_cols[1].checkbox("Apply shape.")
        if use_shape:
            betas = item['shapes']
    # render the pose under the desired viewpoint, and display it
    view_angle = st.slider("Point of view:", min_value=-180, max_value=180, step=20, value=0)
    viewpoint = [] if view_angle == 0 else (view_angle, (0,1,0))
    # ... produce the image!
    pose_img = demo.pose_to_image(pose, body_model, mv, betas=betas, code_base="smplx", viewpoint=viewpoint)
    pose_img_cols[0].image(pose_img, caption=f"Pose")

# text
st.write("**Annotated text:**")
theresmore, text_index = True, 0
while theresmore:
    try:
        st.write(f"_{dataset.get_raw_text(data_id, text_index)}_")
        text_index+=1
    except IndexError:
        theresmore = False