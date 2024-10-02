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
from body_visualizer.mesh.mesh_viewer import MeshViewer

from text2pose.encoders.tokenizers import Tokenizer

import poseembroider.demo as demo
from poseembroider.datasets.bedlam_script import BEDLAMScript
from poseembroider.datasets.threedpw import ThreeDPW
from poseembroider.augmentations import DataProcessingModule
from poseembroider.evaluate_core import load_model, infer_collection_features

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

# place the testing/validation splits first, for default selection
available_data = {
	"bedlamscript-overlap30_res224_j16_sf11": ["val", "train"],
	"threedpw-sf1": ["test", "val"],
	}


### SETUP
################################################################################

# --- models
model, tokenizer_name = demo.setup_models([args.model_path], args.checkpoint, load_model, device=COMPUTING_DEVICE)
model, tokenizer_name = model[0], tokenizer_name[0]
tokenizer = Tokenizer(tokenizer_name)
data_processing = DataProcessingModule(phase="eval") # include image processing

body_model = demo.setup_body_model()
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
						num_body_joints=model.pose_encoder.num_body_joints,
						tokenizer_name=tokenizer_name,
						text_index=0,
						load_raw_texts=True)
		# change the order the poses to display most interesting & diverse poses
		# first
		dataset.change_order_to_farther_sampling_order()
	elif "threedpw" in dataset_version:
		dataset = ThreeDPW(version=dataset_version,
						   split=split_for_research,
						   num_body_joints=model.pose_encoder.num_body_joints,
						   item_format='ip')
	else:
		raise NotImplementedError
	
	return dataset


# --- viewer
imw, imh = 224, 224 # increase size for better resolution
mv = MeshViewer(width=imw, height=imh, use_offscreen=True)
RENDERING_COLOR=[0.4, 0.8, 1.0]


### UTILS
################################################################################

@st.cache_data
def get_features(_model, _dataset): # underscore before variable names to tell Streamlit not to hash them
	# return dict with image/pose/text features
	print("Caching features for faster retrieval... make take a bit of time...")
	return infer_collection_features(_model, _dataset, device=COMPUTING_DEVICE)


def infer(features, item, query_modalities):
	
	# get the subset of available modalities for retrieval from the dataset
	# setting
	retrieved_modalities = []
	if 'i' in dataset.item_format: retrieved_modalities.append('image')
	if 'p' in dataset.item_format: retrieved_modalities.append('pose')
	if 't' in dataset.item_format: retrieved_modalities.append('text')
	
	# init results
	relevant_indices = {m:[] for m in retrieved_modalities}

	# load & prepare data
	item = {k:(torch.tensor([v]) if k=='texts_lengths' else v.unsqueeze(0)) for k,v in item.items() if k not in ["indices", "dataset", "data_ids"]}
	if 'texts_tokens' in item:
		item["texts_tokens"] = item["texts_tokens"][:,:item["texts_lengths"]] # truncate within the batch, based on the longest text 
	item = data_processing(item) # process images
	item = {k:v.to(COMPUTING_DEVICE) if k not in ["indices", "dataset"] else v for k,v in item.items() } # set on device
	
	if model.__class__.__name__ == "PoseEmbroider":

		# get modality projections (query features)
		x_proj, _ = model.get_query_features(**item, query_modalities=query_modalities)
		
		for m_r in retrieved_modalities:
			# rank m_r features (target) by relevance to the m_r projection
			# (query), and get their indices 
			scores = x_proj[f'predicted_{m_r}'].view(1,-1).mm(features[m_r].t())[0]
			_, indices_rank = scores.sort(descending=True)
			relevant_indices[m_r] = indices_rank[:args.n_retrieve].cpu().tolist()

	elif model.__class__.__name__ == "Aligner":

		# get query feature
		query_feature = model.get_query_features(**item, query_modalities=query_modalities)

		for m_r in retrieved_modalities:
			# compute score based on the query feature
			scores = query_feature.mm(features[m_r].t())[0]
			# rank m_r by relevance and get their indices 
			_, indices_rank = scores.sort(descending=True)
			relevant_indices[m_r] = indices_rank[:args.n_retrieve].cpu().tolist()
		
	return relevant_indices


### MAIN APP
################################################################################

print("#####") # to distinguish logs related to each sample

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


# FEATURE COMPUTATION
# -------------------
features = get_features(model, dataset) # caching
print("(features are cached)")


# RETRIEVAL
# ---------

query_modalities = []
if use_image: query_modalities += ["image"]
if use_pose: query_modalities += ["pose"]
if use_text: query_modalities += ["text"]
if use_input_text:
	assert not use_text, "The input text overrides the inital annotated text."
	item["texts_tokens"] = tokenizer(input_text)
	item["texts_lengths"] = len(item["texts_tokens"])
	query_modalities += ["text"]


# retrieve & predict pose if applicable
relevant_indices_per_m = {}
if len(query_modalities):
	relevant_indices_per_m = infer(features, item, query_modalities)


# DISPLAY RESULTS
# ---------------

if query_modalities: # do nothing if no input is selected

	print(relevant_indices_per_m)
	for m, selected_indices in relevant_indices_per_m.items():
		st.write(f"**Relevant {m}s**:")
		if m == "image":
			img_cols = st.columns(4)
			for i, si in enumerate(selected_indices):
				img_cols[i%4].image(demo.convert_img_for_st(dataset.get_image(dataset.index_2_id_list[si])))
		elif m == "pose":
			pose_cols = st.columns(4)
			for i, si in enumerate(selected_indices):
				pose_cols[i%4].image(demo.pose_to_image(dataset.get_pose(dataset.index_2_id_list[si]), body_model, mv, code_base="smplx"))
		elif m == "text":
			for i, si in enumerate(selected_indices):
				text_cols = st.columns([3,1])
				text_cols[0].write(f"**{i+1})** _{dataset.get_raw_text(dataset.index_2_id_list[si])}_")
				text_cols[1].image(demo.pose_to_image(dataset.get_pose(dataset.index_2_id_list[si]), body_model, mv, code_base="smplx"))


st.markdown("---")
st.write(f"**Model:** _{args.model_path}_")