##############################################################
## poseembroider                                            ##
## Copyright (c) 2024                                       ##
## Institut de Robotica i Informatica Industrial, CSIC-UPC  ##
## and Naver Corporation                                    ##
## Licensed under the CC BY-NC-SA 4.0 license.              ##
## See project root for license details.                    ##
##############################################################
 
import torch
import torchvision.transforms as transforms
from copy import deepcopy
import roma

import poseembroider.config as config
from text2pose.posescript.utils import ALL_JOINT_NAMES


# Joint augmentation module
################################################################################

class DataProcessingModule():
	"""
	Expects batch input.
	"""

	def __init__(self, phase="train",
					nb_joints=config.NB_INPUT_JOINTS,
			  		lr_flip_proba=0.5,
					no_img_augmentations=False,
					img_processing_scheme='smplerx',
					tokenizer=None):
		super(DataProcessingModule, self).__init__()
		
		assert phase in ["train", "eval"]
		self.phase = phase

		# initialize image augmentation/processing modules
		acting_phase_imgproc = "eval" if no_img_augmentations else phase
		if img_processing_scheme is None:
			print("Not initializing the image augmentation pipeline.")
		else:
			self.image_transformation = get_image_transformation(acting_phase_imgproc, img_processing_scheme)

		# initialize pose & text augmentation/processing modules
		if self.phase == "train":	
			# L/R flip
			self.lr_flip_proba = lr_flip_proba
			if self.lr_flip_proba:
				self.poseflip = PoseFlip(nb_joints)
				self.tokenizer = tokenizer


	def apply_LR_flip(self, batch_items, lr_flip_proba=None):

		# if not given a flipping proba, use the default one
		if lr_flip_proba is None:
			lr_flip_proba = self.lr_flip_proba

		if lr_flip_proba:

			update_dict = {}

			batch_size = len(batch_items[list(batch_items.keys())[0]])

			# flip text --> determines which elements can be flipped
			flippable = torch.rand(batch_size) < lr_flip_proba
			if 'texts_tokens' in batch_items:
				texts_tokens = batch_items["texts_tokens"]
				texts_lengths = batch_items["texts_lengths"]
				texts_tokens, texts_lengths, flipped = self.tokenizer.flip(texts_tokens, flippable)
				update_dict.update(dict(
					texts_tokens=texts_tokens,
					texts_lengths=texts_lengths
				))
			else:
				flipped = flippable

			# flip images
			for k in ["images", "images_A", "images_B"]:
				if k in batch_items:
					images = batch_items[k]
					images[flipped] = transforms.RandomHorizontalFlip(1)(images[flipped])
					update_dict[k] = images

			# flip camera rotations
			if "cam_rot" in batch_items:
				cam_rot = batch_items["cam_rot"]
				cam_rot[flipped] = roma.rotvec_inverse(cam_rot[flipped])
				update_dict["cam_rot"] = cam_rot

			# flip poses
			for k in ["poses", "poses_A", "poses_B"]:
				if k in batch_items:
					poses = batch_items[k]
					poses[flipped] = self.poseflip(poses[flipped])
					update_dict[k] = poses

			# update
			batch_items.update(update_dict)
			batch_items['flipped'] = flipped

		return batch_items


	def __call__(self, batch_items, lr_flip_proba=None):

		if self.phase == "train":
			batch_items = self.apply_LR_flip(batch_items, lr_flip_proba)

		for k in ["images", "images_A", "images_B"]:
			if k in batch_items:
				batch_items[k] = self.image_transformation(batch_items[k])

		return batch_items


# Pose augmentations
################################################################################

class PoseFlip():

	def __init__(self, nb_joints=22):
		super(PoseFlip, self).__init__()

		# get joint names (depends on the case)
		if nb_joints == 21:
			# all main joints, without the root
			joint_names = ALL_JOINT_NAMES[1:22]
		elif nb_joints == 22:
			# all main joints, with the root
			joint_names = ALL_JOINT_NAMES[:22]
		elif nb_joints == 52:
			joint_names = ALL_JOINT_NAMES[:]
		else:
			raise NotImplementedError

		# build joint correspondance indices
		n2i = {n:i for i, n in enumerate(joint_names)}
		l2r_j_id = {i:n2i[n.replace("left", "right")] for n,i in n2i.items() if "left" in n} # joint index correspondance between left and right
		self.left_joint_inds = torch.tensor(list(l2r_j_id.keys()))
		self.right_joint_inds = torch.tensor(list(l2r_j_id.values()))

	def flip_pose_data_LR(self, pose_data):
		"""
		pose_data: shape (batch_size, nb_joint, 3)
		"""
		l_data = deepcopy(pose_data[:,self.left_joint_inds])
		r_data = deepcopy(pose_data[:,self.right_joint_inds])
		pose_data[:,self.left_joint_inds] = r_data
		pose_data[:,self.right_joint_inds] = l_data
		pose_data[:,:, 1:3] *= -1
		return pose_data
	
	def __call__(self, pose_data):
		return self.flip_pose_data_LR(pose_data.clone())


# Transformation functions
################################################################################

def rescale_values():
	"""
	Returns a transformation function that expects a uint tensor and rescales
	its values between 0 and 1.
	The clamping operation accounts for previous transformations that may have
	produced out-of-range values.
	"""
	return lambda x: x.clamp(0, 255).to(torch.float32).div(255.)


class ElementWise:
	"""
	Apply a transformation to each element of the batch independently.
	"""
	def __init__(self, transformation):
		self.transformation = transformation

	def __call__(self, batch):
		return torch.stack([self.transformation(batch[i]) for i in range(len(batch))])


def transform_for_visu(
		unormalize=True,
		mean=config.IMAGENET_DEFAULT_MEAN,
		std=config.IMAGENET_DEFAULT_STD
	):
	"""
	Unormalize processed images for visualization.
	"""
	# after images have been processed with torchvision transforms, there might
	# be some adjustments to make to visualize data with streamlit (eg. undo
	# ImageNet normalization)
	tlist = []
	if unormalize:
		tlist.append(transforms.Normalize(
			   mean= [-m/s for m, s in zip(mean, std)],
			   std= [1/s for s in std]
			))
	return transforms.Compose(tlist)


# Image transformation getter
################################################################################

# NOTE: do not use horizontal flip or rotation augmentations in training
# transformations as there is inter-modality dependence.


def get_image_transformation(phase, scheme='smplerx'):
	p = 'training' if phase=='train' else 'evaluation'
	return eval(f"get_img_{scheme}_{p}_transformation()")


def get_img_smplerx_training_transformation(
		crops_size=(256, 192),
		scale=(0.5,1.0),
		radius_min=0.1,
		radius_max=1.0,
		patch_random_blur_proba=0.1,
		interpolation=transforms.InterpolationMode.BICUBIC
	):
	# Expects integer tensors.
	return ElementWise(transforms.Compose([

		# geometric transformations
		transforms.RandomResizedCrop(
			crops_size, scale=scale, interpolation=interpolation, antialias=True
		),

		# color jittering
		transforms.RandomApply(
			[transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
			p=0.8,
		),

		# gaussian blur
		transforms.RandomApply(
			[transforms.GaussianBlur(kernel_size=9, sigma=(radius_min, radius_max))],
			p=1-patch_random_blur_proba
		),
		transforms.RandomSolarize(threshold=128, p=0.2),
		
		# normalization
		rescale_values()
	]))


def get_img_smplerx_evaluation_transformation():
	return ElementWise(transforms.Compose([rescale_values()]))