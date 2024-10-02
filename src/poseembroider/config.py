##############################################################
## poseembroider                                            ##
## Copyright (c) 2024                                       ##
## Institut de Robotica i Informatica Industrial, CSIC-UPC  ##
## and Naver Corporation                                    ##
## Licensed under the CC BY-NC-SA 4.0 license.              ##
## See project root for license details.                    ##
##############################################################

# default
import os
MAIN_DIR = os.path.realpath(__file__)
MAIN_DIR = os.path.dirname(os.path.dirname(os.path.dirname(MAIN_DIR)))


################################################################################
# Output dir for experiments
################################################################################

GENERAL_EXP_OUTPUT_DIR = MAIN_DIR + '/experiments'


################################################################################
# DATA LOCATIONS
################################################################################

# Datasets
BEDLAM_RAW_DATA_DIR = MAIN_DIR + "data/BEDLAM"
# NOTE: required subdirectories:
# * neutral_ground_truth_motioninfo (visit BEDLAM website, and download the SMPL-X neutral ground truth/animation files)
# * csv (visit BEDLAM website, and download "be_imagedata_download.zip", the "csv" directory is inside it)
BEDLAM_IMG_DIR = MAIN_DIR + "data/BEDLAM/images"
BEDLAM_PROCESSED_DIR = MAIN_DIR + "data/BEDLAM/processed"

THREEDPW_IMG_DIR = MAIN_DIR + "data/threedpw"
THREEDPW_DIR = MAIN_DIR + "data/threedpw"

POSEFIX_DIR = MAIN_DIR + "/data/PoseFix"
POSEFIX_SMPLX_DIR = MAIN_DIR + "/data/PoseFix/output_data_pkl"

POSESCRIPT_DIR = None
POSESCRIPT_SMPLX_DIR = POSEFIX_SMPLX_DIR

# Body models
SMPLX_BODY_MODEL_PATH = MAIN_DIR + "/tools/smpl_models"
# NOTE: SMPLX_BODY_MODEL_PATH should be a directory with "./smplx/SMPLX_NEUTRAL.npz" in it

# Cached data
DATA_CACHE_DIR = MAIN_DIR + "/cache"

# Pretrained models
PRETRAINED_MODEL_DICT = MAIN_DIR + "/src/poseembroider/shortname_2_model_path.json"


################################################################################
# TOOLS LOCATIONS
################################################################################

TORCH_CACHE_DIR = MAIN_DIR + '/tools/torch_models' # should contain a folder 'smplerx' with file 'smpler_x_b32.pth.tar'
TRANSFORMER_CACHE_DIR = MAIN_DIR + '/tools/huggingface_models' # should contain 'distilbert-base-uncased'
MEAN_SMPLX_POSE_FILE = MAIN_DIR + '/src/poseembroider/datasets/other/mean_pose_smplx_bedlam.pkl'


################################################################################
# DATA FORMAT & PROCESSING
################################################################################

NB_INPUT_JOINTS=22
NB_SHAPE_COEFFS=10

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


################################################################################
# EVALUATION
################################################################################

# -- (specific to PoseEmbroider/Aligner)
k_recall_values = [1, 5, 10]
# -- (specific to InstructionGenerator)
k_topk_r_precision = [1,2,3]
sample_size_r_precision = 200
r_precision_n_repetitions = 10
# -- (specific to PoseVAE)
k_topk_reconstruction_values = [1]
nb_sample_reconstruction = 30