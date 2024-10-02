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
import roma
from body_visualizer.mesh.mesh_viewer import MeshViewer

from text2pose.encoders.tokenizers import Tokenizer

import poseembroider.demo as demo
import poseembroider.utils as utils
from poseembroider.datasets.bedlam_script import BEDLAMScript
from poseembroider.datasets.threedpw import ThreeDPW
from poseembroider.augmentations import DataProcessingModule
from poseembroider.downstream.pose_estimation.evaluate_pose_estimation import load_model
from poseembroider.evaluator import procrustes_align


parser = argparse.ArgumentParser(description='Parameters for the demo.')
parser.add_argument('--model_path', type=str, help='Path to the model.')
parser.add_argument('--checkpoint', default='best', choices=('best', 'last'), help='Checkpoint to choose if model path is incomplete.')
parser.add_argument('--n_retrieve', type=int, default=8, help="Number of elements to retrieve.")
args = parser.parse_args()


### SETTINGS
################################################################################

COMPUTING_DEVICE = torch.device('cuda:0')

IMG_SIZE=(256, 192) # from SMPLer-X
PATCH_SIZE=16 # from SMPLer-X  

# set the testing/validation splits first, for default selection
available_data = {
	"bedlamscript-overlap30_res224_j16_sf11": ["val", "train"],
	"threedpw-sf1": ["test", "val"],
	}


### SETUP
################################################################################

# --- models
model, tokenizer_name = demo.setup_models([args.model_path], args.checkpoint, load_model, device=COMPUTING_DEVICE)
model, tokenizer_name = model[0], tokenizer_name[0]
num_betas = model.num_shape_coeffs
num_joints = model.num_body_joints
tokenizer = Tokenizer(tokenizer_name)
img_processing_scheme = utils.get_img_processing_scheme(model.representation_wrapper.representation_model.image_encoder_name)
data_processing = DataProcessingModule(phase="eval", img_processing_scheme=img_processing_scheme) # include image processing

body_model = demo.setup_body_model(num_betas=num_betas)
body_model.eval()
body_model.to('cpu')


# --- data
@st.cache_resource
def get_data(dataset_version, split_for_research):
	if "bedlamscript" in dataset_version:
		fs_size = {"val":10000, "train":50000}[split_for_research]
		dataset = BEDLAMScript(version=dataset_version,
						split=split_for_research,
						reduced_set=fs_size,
						num_body_joints=num_joints,
						num_shape_coeffs=num_betas,
						img_processing_scheme=img_processing_scheme,
						tokenizer_name=tokenizer_name,
						text_index=0,
						load_raw_texts=True)
		# change the order the poses to display most interesting & diverse poses
		# first
		dataset.change_order_to_farther_sampling_order()
	elif "threedpw" in dataset_version:
		dataset = ThreeDPW(version=dataset_version,
						   split=split_for_research,
						   num_body_joints=num_joints,
						   num_shape_coeffs=num_betas,
						   img_processing_scheme=img_processing_scheme,
						   item_format='ip')
	else:
		raise NotImplementedError
	
	return dataset


# --- viewer
imw, imh = 224, 224 # increase size for better resolution
mv = MeshViewer(width=imw, height=imh, use_offscreen=True)
RENDERING_COLOR=[0.4, 0.8, 1.0]


### MAIN APP
################################################################################

print("#####")

# QUERY SELECTION
# ---------------
cols_query = st.columns(2)

# choose dataset
dataset_version = cols_query[0].selectbox("Dataset:", list(available_data.keys()), index=0)
split_for_research = cols_query[1].selectbox("Split:", available_data[dataset_version], index=0)
# load dataset
dataset = get_data(dataset_version, split_for_research)

# define query input interface: example selection
query_type = cols_query[0].selectbox("Query type:", ('Split index', 'ID'))
number = cols_query[1].number_input("Split index or ID:", 0, len(dataset.index_2_id_list))
if query_type == "ID":
	print("### Examples of available data IDs:", dataset.index_2_id_list[:10])

# get data_id / item_index
data_id = number if query_type == 'ID' else dataset.index_2_id_list[number]
item_index = dataset.index_2_id_list.index(number) if query_type == 'ID' else number

# get query data
item = dataset.__getitem__(item_index)
raw_text = dataset.get_raw_text(data_id, 0) if 't' in dataset.item_format else ''


# QUERY DISPLAY
# -------------

st.write(f"**Data ID:** {data_id}")

# image masking
if 'i' in dataset.item_format:
	mask_x = st.slider("Mask position (x):", min_value=0, max_value=IMG_SIZE[1], value=0, step=int(IMG_SIZE[1]/PATCH_SIZE))
	mask_y = st.slider("Mask position (y):", min_value=0, max_value=IMG_SIZE[0], value=0, step=int(IMG_SIZE[0]/PATCH_SIZE))
	mask_scale = st.slider("Scale (patches):", min_value=0, max_value=int(max(IMG_SIZE)/PATCH_SIZE), value=0, step=1)
	x1,y1,x2,y2 = mask_x, mask_y, min(mask_x+mask_scale*PATCH_SIZE, IMG_SIZE[1]), min(mask_y+mask_scale*PATCH_SIZE, IMG_SIZE[0])
	item["images"][:,y1:y2,x1:x2] = 0

# form initialization
form = st.form("query_form")
form.write("Select what to use as input.")
cols_input = form.columns(2)

# image
if 'i' in dataset.item_format:
	use_image = cols_input[0].checkbox("**Use the image!**")
	cols_input[0].image(demo.convert_img_for_st(item['images']), caption="Image")
else:
	use_image = False

# pose
if 'p' in dataset.item_format:
	use_pose = cols_input[1].checkbox("**Use the pose!**")
	pose_img = demo.pose_to_image(item['poses'], body_model, mv, betas=item.get("shapes", None), color='purple', code_base="smplx")
	cols_input[1].image(pose_img, caption="Annotated pose")
else:
	use_pose = False

# text
if 't' in dataset.item_format:
	use_text = form.checkbox("**Use the annotated text!**")
	form.write(f"_{raw_text}_")
else:
	use_text = False

use_input_text = form.checkbox("**Use the input text!**")
input_text = form.text_area("Pose description:",
								value=raw_text,
								placeholder="The person is...",
								height=None, max_chars=None)

# add the form submit button
submit_input = form.form_submit_button("Process!")


# INFER
# -----

# define model inputs
representation_model_input_types = []
if use_image: representation_model_input_types += ["images"]
if use_pose: representation_model_input_types += ["poses"]
if use_text: representation_model_input_types += ["texts_tokens", "texts_lengths"]
if use_input_text:
	assert not use_text, "The input text overrides the inital annotated text."
	item["texts_tokens"] = tokenizer(input_text)
	item["texts_lengths"] = len(item["texts_tokens"])
	representation_model_input_types += ["texts_tokens", "texts_lengths"]

if 'texts_lengths' in item:
	# since there is only one sample, we need to "tensorize" the text length
	item["texts_lengths"] = torch.tensor([item["texts_lengths"]])

# predict pose & shape
if representation_model_input_types: # do nothing if no input is selected
	predicted_pose = {}
	with torch.no_grad() and torch.inference_mode():

		# prepare data
		item = {k:v.to(COMPUTING_DEVICE).unsqueeze(0) if type(v)==torch.Tensor else [v] for k,v in item.items() } # set on device + batchify
		item = data_processing(item) # image processing
		if 'texts_tokens' in item:
			# truncate within the batch, based on the longest text 
			item["texts_tokens"] = item["texts_tokens"][:,:max(item["texts_lengths"])]

		# infer
		pred_rotmat, pred_betas = model(item=item, representation_model_input_types=representation_model_input_types) # `pred_rotmat`: shape (n_samples, n_joints, 3, 3) ; `pred_betas`: (n_samples, n_betas) or None
		predicted_pose["joint_rots"] = roma.rotmat_to_rotvec(pred_rotmat) # shape (n_samples, n_joints, 3)
		predicted_pose["shape"] = pred_betas

	# set back on cpu + unbatchify
	item = {k:v.to("cpu")[0] if type(v)==torch.Tensor else v for k,v in item.items() }
	predicted_pose = {k:v.to('cpu')[0] if v is not None else v for k,v in predicted_pose.items()}


# DISPLAY RESULTS
# ---------------

if representation_model_input_types: # do nothing if no input is selected

	# choose a viewpoint for rendering
	view_angle = st.slider("Point of view:", min_value=-180, max_value=180, step=20, value=0)
	viewpoint = [] if view_angle == 0 else (view_angle, (0,1,0))

	# display the predicted pose, along with the ground truth pose if available;
	# in the latter case, propose to modify the global rotation with respect to
	# the camera by applying procrustes alignment
	if 'p' in dataset.item_format:
		apply_pa = st.checkbox("Align procrustes in camera reference frame.")
		pose_pred_cols = st.columns(2)
		if apply_pa:
			# get the GT pose in the camera reference frame
			gt_camera_pose = item['poses'].clone()
			gt_camera_pose[0:1] = roma.rotvec_composition([item['cam_rot'].view(1,3), gt_camera_pose[0:1].clone()])
			# utils for procrustes alignment
			joint_set = torch.arange(22)
			rs = lambda v: v.view(1,-1) if v is not None else None 
			# get joint positions
			pred_pose_j = body_model(**utils.pose_data_as_dict(predicted_pose["joint_rots"], code_base="smplx"), betas=rs(predicted_pose["shape"])).joints[:,joint_set]
			gt_pose_j = body_model(**utils.pose_data_as_dict(gt_camera_pose, code_base="smplx"), betas=rs(item.get("shapes", None))).joints[:,joint_set]
			# compute rotation for procrustes alignment
			pred_R,_,_ = procrustes_align(pred_pose_j, gt_pose_j, return_transformation=True)
			# alterate the global orientation to account for procrustes alignment
			pred_pose = predicted_pose["joint_rots"].clone()
			pred_pose[0] = roma.rotvec_composition([roma.rotmat_to_rotvec(pred_R[0]), pred_pose[0].clone()])
			# display
			pose_pred_cols[0].image(demo.pose_to_image(pred_pose, body_model, mv, betas=predicted_pose["shape"], code_base="smplx", viewpoint=viewpoint), caption="Predicted pose")
			pose_pred_cols[1].image(demo.pose_to_image(gt_camera_pose, body_model, mv, color="purple", betas=item.get("shapes", None), code_base="smplx", viewpoint=viewpoint), caption="GT pose & shape")
		else:
			pose_pred_cols[0].image(demo.pose_to_image(predicted_pose["joint_rots"], body_model, mv, betas=predicted_pose["shape"], code_base="smplx", viewpoint=viewpoint), caption="Predicted pose")
			pose_pred_cols[1].image(demo.pose_to_image(item['poses'], body_model, mv, color="purple", betas=item.get("shapes", None), code_base="smplx", viewpoint=viewpoint), caption="GT pose & shape")
	else:
		# just show the predicted pose
		st.image(demo.pose_to_image(predicted_pose["joint_rots"], body_model, mv, betas=predicted_pose["shape"], code_base="smplx", viewpoint=viewpoint), caption="Predicted pose")