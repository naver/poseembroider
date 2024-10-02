##############################################################
## poseembroider                                            ##
## Copyright (c) 2024                                       ##
## Institut de Robotica i Informatica Industrial, CSIC-UPC  ##
## and Naver Corporation                                    ##
## Licensed under the CC BY-NC-SA 4.0 license.              ##
## See project root for license details.                    ##
##############################################################

"""
The process is composed of the following steps:

1. Extract relevant pose frames from AMASS sequences.
   ie. poses belonging to PoseScript/PoseFix.

   NOTE: this process could be even more optimized by considering together all
   poses belonging to the same sequence. We don't do it here for code simplicity.

2. Process the poses with the SMPL-H body model, to get corresponding meshes
   (.obj files). Meshes are obtained using the neutral body model and the
   default shape. Each mesh is to be saved indivually.

NOTE: in the following, one can choose to generate the mesh using either the
body model from the AMASS codebase (ANNOTATION_MODE = True), in the same way it
was shown to the annotators; or using the smplx codebase (ANNOTATION_MODE =
False), similarly to the initial version of this code. The same thing can be
achieved with both modes.
"""

# IMPORTS
################################################################################

import os
import numpy as np
import torch
import trimesh
import smplx
from human_body_prior.body_model.body_model import BodyModel
from tqdm import tqdm


# SETUP
################################################################################

ANNOTATION_MODE = True # (see note at the top)


# TODO: AMASS_BODY_MODEL_AMASS_CODEBASE corresponds to the zip from "Extended
# SMPL+H model" link on the official AMASS website, the path must finish by
# "neutral/model.npz" ; alternatively, if ANNOTATION_MODE is False, you can use
# AMASS_BODY_MODEL_SMPLX_CODEBASE instead (see README), which is a directory
# that should contain "smplh/SMPLH_NEUTRAL.pkl".
AMASS_BODY_MODEL_AMASS_CODEBASE = None
AMASS_BODY_MODEL_SMPLX_CODEBASE = None
# TODO: AMASS_FILE_LOCATION_SMPLH corresponds to the location of the AMASS
# sequences, in SMPL-H format
AMASS_FILE_LOCATION_SMPLH = None
# TODO: OBJ_OUTPUT_DIR is your preferred output location (to be used as input to
# the conversion algorithm)
OBJ_OUTPUT_DIR = None
# TODO: POSEFIX_DIR & POSESCRIPT_DIR are the locations of both datasets
POSEFIX_DIR = None
POSESCRIPT_DIR = None # (OPTIONAL) if you fill this field, PoseScript poses will be processed as well

# sanity check
if POSEFIX_DIR is None \
	or OBJ_OUTPUT_DIR is None \
    or (AMASS_BODY_MODEL_AMASS_CODEBASE is None and AMASS_BODY_MODEL_SMPLX_CODEBASE is None) \
	or AMASS_FILE_LOCATION_SMPLH is None:
	print(f"Please complete the fields noted with TODO at the top of this file ({os.path.realpath(__file__)}).")
	import sys
	sys.exit(-1)


# READ/WRITE TO FILES
################################################################################

import json

def read_json(absolute_filepath):
	with open(absolute_filepath, 'r') as f:
		data = json.load(f)
	return data

def write_json(data, absolute_filepath, pretty=True):
	with open(absolute_filepath, "w") as f:
		if pretty:
			json.dump(data, f, ensure_ascii=False, indent=2)
		else:
			json.dump(data, f)


# MAIN
################################################################################

# initialize body model
if ANNOTATION_MODE:
	body_model = BodyModel(model_type="smplh",
						   bm_fname=AMASS_BODY_MODEL_AMASS_CODEBASE,
						   num_betas=16
						)
else:
	body_model = smplx.create(AMASS_BODY_MODEL_SMPLX_CODEBASE,
							model_type="smplh",
							gender="neutral",
							num_betas=10,
							use_pca=False,
							flat_hand_mean = True, # ADDED!
							ext="pkl",
						)
body_model.eval()

# initialize vertex color variable
n_vertices = 6890 # SMPL-H
vertex_colors = np.ones([6890, 4]) * [0.3, 0.3, 0.3, 0.8]

# create saving directory
os.makedirs(OBJ_OUTPUT_DIR, exist_ok=True)

# load pose info
dataID_2_pose_info = read_json(os.path.join(POSEFIX_DIR, "ids_2_dataset_sequence_and_frame_index_100k.json"))

# define priority IDs
posefix_H_pair_ids = list(read_json(os.path.join(POSEFIX_DIR, "posefix_human_6157.json")).keys())
posefix_pair_id_2_pose_ids = read_json(os.path.join(POSEFIX_DIR, "pair_id_2_pose_ids.json"))
posefix_H_pose_ids = [posefix_pair_id_2_pose_ids[int(pair_id)] for pair_id in posefix_H_pair_ids] # list of lists
posefix_H_pose_ids = [str(x) for xs in posefix_H_pose_ids for x in xs] # flatten list + set it to str

if POSESCRIPT_DIR is not None:
    posescript_H_pose_ids = list(read_json(os.path.join(POSESCRIPT_DIR, "posescript_human_6293.json")).keys())
    priority_ids = list(set(posescript_H_pose_ids + posefix_H_pose_ids)) # consider poses both in PoseScript-H and PoseFix-H
else:
    priority_ids = list(set(posefix_H_pose_ids)) 	

print(f"Priority poses: {len(priority_ids)} poses.")

# process each pose one by one
for data_id in tqdm(dataID_2_pose_info.keys()):

	# get pose info
	pose_info = dataID_2_pose_info[data_id]

	# NOTE: limit to priority poses
	if data_id not in priority_ids:
		continue

	# load pose sequence
	dp = np.load(os.path.join(AMASS_FILE_LOCATION_SMPLH, pose_info[1]))
	
	# select the frame of interest
	pose = dp['poses'][pose_info[2],:] # (n_joints * 3)
	pose = torch.as_tensor(pose).view(1,-1).float() # .to(dtype=torch.float32)

	# feed the pose to the body model
	if ANNOTATION_MODE:
		output = body_model(
					root_orient=pose[:,:3],
					pose_body=pose[:,3:66],
					pose_hand=pose[:,66:]
				)
	else:
		output = body_model(
					betas = torch.zeros(1, 10),
					global_orient = pose[:,:3],
					body_pose = pose[:,3:66],
					left_hand_pose = pose[:,66:111],
					right_hand_pose = pose[:,111:156],
					expression = None,
					return_verts = True
				)

	# infer mesh
	vertices = output.v if ANNOTATION_MODE else output.vertices
	mesh = trimesh.Trimesh(
		vertices.detach().cpu().numpy().squeeze(),
		body_model.f if ANNOTATION_MODE else body_model.faces,
		vertex_colors=vertex_colors,
		process=False
	)

	# save mesh as .obj file
	output_path = os.path.join(OBJ_OUTPUT_DIR, "{0:06d}.obj".format(int(data_id)))
	mesh.export(str(output_path))