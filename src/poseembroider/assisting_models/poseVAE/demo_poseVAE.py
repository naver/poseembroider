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

import poseembroider.demo as demo
from poseembroider.datasets.bedlam_script import BEDLAMScript
from poseembroider.assisting_models.poseVAE.evaluate_poseVAE import load_model

parser = argparse.ArgumentParser(description='Parameters for the demo.')
parser.add_argument('--model_paths', nargs='+', type=str, help='Paths to the models to be compared.')
parser.add_argument('--checkpoint', default='best', choices=('best', 'last'), help="Checkpoint to choose if model path is incomplete.")
parser.add_argument('--n_generate', type=int, default=12, help="Number of poses to generate (number of samples); if considering only one model.")
args = parser.parse_args()


### SETTINGS
################################################################################

COMPUTING_DEVICE = torch.device('cpu')

available_data = {"bedlamscript-overlap30_res224_j16_sf11": ["val", "train"]}


### SETUP
################################################################################

# --- models
models, _ = demo.setup_models(args.model_paths, args.checkpoint, load_model, device=COMPUTING_DEVICE)
assert len(set([m.pose_encoder.num_body_joints for m in models])) == 1, "Comparing models using different numbers of joints to represent the pose. Case not handled here. Make the necessary modifications first."
num_body_joints = models[0].pose_encoder.num_body_joints

body_model = demo.setup_body_model()
body_model.eval()
body_model.to('cpu')

# --- data
@st.cache_resource
def get_data(data_version, split_for_research=None):
	if "bedlam" in data_version:
		fs_size = {"val":10000, "train":50000}[split_for_research]
		dataset = BEDLAMScript(version=data_version,
						split=split_for_research,
						reduced_set=fs_size,
						num_body_joints=num_body_joints,
						tokenizer_name=None,
						item_format='p')
		dataset.change_order_to_farther_sampling_order()
	else:
		raise NotImplementedError
	
	return dataset

# --- viewer
imw, imh = 400, 400
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

# correct the number of generated sample depending on the setting
if len(args.model_paths) > 1:
    n_generate = 4
else:
    n_generate = args.n_generate

# --- seed
torch.manual_seed(42)
np.random.seed(42)


### MAIN APP
################################################################################

# QUERY SELECTION
# ---------------

# define query input interface
cols_query = st.columns(2)

# choose dataset
dataset_version = cols_query[0].selectbox("Dataset:", list(available_data.keys()), index=0)
split = cols_query[1].selectbox("Split:", available_data[dataset_version], index=0)
# load dataset
d = get_data(dataset_version, split)

# define query input interface: example selection
query_type = cols_query[0].selectbox("Query type:", ('Split index', 'ID'))
number = cols_query[1].number_input("Split index or ID:", 0, len(d.index_2_id_list))
st.markdown("""---""")

# get data_id / item_index
data_id = number if query_type == 'ID' else d.index_2_id_list[number]
item_index = d.index_2_id_list.index(number) if query_type == 'ID' else number

# get query data
item = d.__getitem__(item_index)


# QUERY DISPLAY
# -------------

cols_input = st.columns(2)
pose_img = demo.pose_to_image(item['poses'], body_model, mv, betas=None, color='purple', code_base="smplx")
cols_input[0].image(pose_img, caption="Annotated pose")
analysis = cols_input[1].checkbox('Analysis') # whether to show the reconstructed pose and the mean sample pose in addition of some samples


# INFER
# -----

# choose a viewpoint for rendering
view_angle = st.slider("Point of view:", min_value=-180, max_value=180, step=20, value=0)
viewpoint = [] if view_angle == 0 else (view_angle, (0,1,0))

# generate results
if analysis:

	st.markdown("""---""")
	st.write("**Generated poses** (*The reconstructed pose is shown in green; the mean pose in red; and samples in grey.*):")
	n_generate = 2
	nb_cols = 2 + n_generate # reconstructed pose + mean sample pose + n_generate sample poses: all must fit in one row, for each studied model

	for i, model in enumerate(models):
		with torch.no_grad():
			rec_pose_data = model.forward(item["poses"].view(1,-1))['pose_body_pose'].view(1, -1)
			gen_pose_data_mean = model.sample_meanposes()['pose_body'].view(1, -1)
			gen_pose_data_samples = model.sample_nposes(n=n_generate)['pose_body'][0,...].view(n_generate, -1)

		# render poses
		imgs = []
		imgs.append(demo.pose_to_image(rec_pose_data, body_model, mv, color="green", betas=None, code_base="smplx", viewpoint=viewpoint), caption="Reconstructed")
		imgs.append(demo.pose_to_image(gen_pose_data_mean, body_model, mv, color="red", betas=None, code_base="smplx", viewpoint=viewpoint), caption="Mean")
		for geni in range(n_generate):
			imgs.append(demo.pose_to_image(gen_pose_data_samples[geni].view(1,-1), body_model, mv, color="grey", betas=None, code_base="smplx", viewpoint=viewpoint), caption="Sample")

		# display images
		cols = st.columns(nb_cols+1) # +1 to display model info
		cols[0].markdown(f'<p class="smallfont">{args.model_paths[i]}</p>', unsafe_allow_html=True)
		for i in range(nb_cols):
			cols[i%nb_cols+1].image(imgs[i])
		st.markdown("""---""")

else:

	st.markdown("""---""")
	st.write("**Generated poses:**")

	for i, model in enumerate(models):
		with torch.no_grad():
			gen_pose_data_samples = model.sample_nposes(n=n_generate)['pose_body'][0,...].view(n_generate, -1)

		# render poses
		imgs = []
		for geni in range(n_generate):
			imgs.append(demo.pose_to_image(gen_pose_data_samples[geni].view(1,-1), body_model, mv, betas=None, code_base="smplx", viewpoint=viewpoint))

		# display images
		if len(models) > 1:
			cols = st.columns(n_generate+1) # +1 to display model info
			cols[0].markdown(f'<p class="smallfont">{args.model_paths[i]}</p>', unsafe_allow_html=True)
			for i in range(n_generate):
				cols[i%n_generate+1].image(imgs[i])
			st.markdown("""---""")
		else:
			cols = st.columns(demo.nb_cols)
			for i in range(n_generate):
				cols[i%demo.nb_cols].image(imgs[i])
			st.markdown("""---""")
			st.write(f"_Results obtained with model: {args.model_paths[0]}_")