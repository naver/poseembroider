import torch

import poseembroider.config as config


################################################################################
## READ/WRITE TO FILES
################################################################################

import json

def read_json(absolute_filepath):
	with open(absolute_filepath, 'r') as f:
		data = json.load(f)
	return data

def write_json(data, absolute_filepath, pretty=False, tell=False):
	with open(absolute_filepath, "w") as f:
		if pretty:
			json.dump(data, f, ensure_ascii=False, indent=2)
		else:
			json.dump(data, f)
	if tell:
		print(f"Saved file: {absolute_filepath}")


import pickle

def read_pickle(absolute_filepath):
	with open(absolute_filepath, 'rb') as f:
		data = pickle.load(f)
	return data

def write_pickle(data, absolute_filepath, tell=False):
	with open(absolute_filepath, 'wb') as f:
		pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
	if tell:
		print(f"Saved file: {absolute_filepath}")


################################################################################
## ANGLE TRANSFORMATION FONCTIONS
################################################################################

import roma

def rotvec_to_eulerangles(x):
	x_rotmat = roma.rotvec_to_rotmat(x)
	thetax = torch.atan2(x_rotmat[:,2,1], x_rotmat[:,2,2])
	thetay = torch.atan2(-x_rotmat[:,2,0], torch.sqrt(x_rotmat[:,2,1]**2+x_rotmat[:,2,2]**2))
	thetaz = torch.atan2(x_rotmat[:,1,0], x_rotmat[:,0,0])
	return thetax, thetay, thetaz

def eulerangles_to_rotmat(thetax, thetay, thetaz):
	N = thetax.numel()
	# rotx
	rotx = torch.eye( (3) ).to(thetax.device).repeat(N,1,1)
	roty = torch.eye( (3) ).to(thetax.device).repeat(N,1,1)
	rotz = torch.eye( (3) ).to(thetax.device).repeat(N,1,1)
	rotx[:,1,1] = torch.cos(thetax)
	rotx[:,2,2] = torch.cos(thetax)
	rotx[:,1,2] = -torch.sin(thetax)
	rotx[:,2,1] = torch.sin(thetax)
	roty[:,0,0] = torch.cos(thetay)
	roty[:,2,2] = torch.cos(thetay)
	roty[:,0,2] = torch.sin(thetay)
	roty[:,2,0] = -torch.sin(thetay)
	rotz[:,0,0] = torch.cos(thetaz)
	rotz[:,1,1] = torch.cos(thetaz)
	rotz[:,0,1] = -torch.sin(thetaz)
	rotz[:,1,0] = torch.sin(thetaz)
	rotmat = torch.einsum('bij,bjk->bik', rotz, torch.einsum('bij,bjk->bik', roty, rotx))
	return rotmat

def eulerangles_to_rotvec(thetax, thetay, thetaz):
	rotmat = eulerangles_to_rotmat(thetax, thetay, thetaz)
	return roma.rotmat_to_rotvec(rotmat)


################################################################################
## POSE DATA FORMATTING & BODY MODEL
################################################################################


def pose_data_as_dict(pose_data, code_base='smplx'):
	"""
	Args:
		pose_data, torch.tensor of shape (*, n_joints*3) or (*, n_joints, 3),
			all joints considered.
	Returns:
		dict
	"""
	# reshape to (*, n_joints*3) if necessary
	if len(pose_data.shape) == 3:
		# shape (batch_size, n_joints, 3)
		pose_data = pose_data.flatten(1,2)
	if len(pose_data.shape) == 2 and pose_data.shape[1] == 3:
		# shape (n_joints, 3)
		pose_data = pose_data.view(1, -1)
	# provide as a dict, with different keys, depending on the code base
	if code_base == 'human_body_prior':
		d = {"root_orient":pose_data[:,:3],
	   		 "pose_body":pose_data[:,3:66]}
		if pose_data.shape[1] > 66:
			d["pose_hand"] = pose_data[:,66:]
	elif code_base == 'smplx':
		d = {"global_orient":pose_data[:,:3],
	   		 "body_pose":pose_data[:,3:66]}
		if pose_data.shape[1] > 66:
			d.update({"left_hand_pose":pose_data[:,66:111],
					"right_hand_pose":pose_data[:,111:]})
	return d


import os
import smplx
from typing import Optional


class BodyModelSMPLX(smplx.SMPLX):
	"""
	At the time of this project, many body models in the smplx codebase expect
	inputs that always have the same batch size.
	This wrapper makes it possible to dynamically adapt to the actual batch size
	of the input.
	"""
	def __init__(self, model_path=config.SMPLX_BODY_MODEL_PATH,
			batch_size: int = 1,
			gender='neutral',
			num_betas=config.NB_SHAPE_COEFFS,
			use_pca=False,
			flat_hand_mean=True,
			**kwargs):
		model_path = os.path.join(model_path, 'smplx')
		self.batch_size = batch_size
		smplx.SMPLX.__init__(self, model_path,
					   			batch_size=batch_size,
								gender=gender,
								num_betas=num_betas,
								use_pca=use_pca,
								flat_hand_mean=flat_hand_mean,
								**kwargs)

	def forward(self,
		betas: Optional[torch.Tensor] = None,
        global_orient: Optional[torch.Tensor] = None,
        body_pose: Optional[torch.Tensor] = None,
        left_hand_pose: Optional[torch.Tensor] = None,
        right_hand_pose: Optional[torch.Tensor] = None,
        transl: Optional[torch.Tensor] = None,
        expression: Optional[torch.Tensor] = None,
        jaw_pose: Optional[torch.Tensor] = None,
        leye_pose: Optional[torch.Tensor] = None,
        reye_pose: Optional[torch.Tensor] = None,
	):
		# get batch size
		batch_size = body_pose.shape[0]
		batch_size_setting = self.batch_size
		# adapt to actual batch size
		# NOTE: using values stored for the first element of the batch as repeated default
		adapt_bs = lambda x: x[0].unsqueeze(0).repeat([(batch_size if i==0 else 1) for i in range(len(x.shape))])
		if batch_size_setting != batch_size:
			betas = betas if betas is not None else adapt_bs(self.betas)
			global_orient = (global_orient if global_orient is not None else
							adapt_bs(self.global_orient))
			body_pose = body_pose if body_pose is not None else adapt_bs(self.body_pose)
			left_hand_pose = (left_hand_pose if left_hand_pose is not None else
							adapt_bs(self.left_hand_pose))
			right_hand_pose = (right_hand_pose if right_hand_pose is not None else
							adapt_bs(self.right_hand_pose))
			expression = expression if expression is not None else adapt_bs(self.expression)
			jaw_pose = jaw_pose if jaw_pose is not None else adapt_bs(self.jaw_pose)
			leye_pose = leye_pose if leye_pose is not None else adapt_bs(self.leye_pose)
			reye_pose = reye_pose if reye_pose is not None else adapt_bs(self.reye_pose)
			transl = transl if transl is not None else (adapt_bs(self.transl) if hasattr(self, 'transl') else None)
			# Note, since the forward function in the library calls both
			# `batch_size` and `self.batch_size`, we have to modify
			# self.batch_size temporarily...
			self.batch_size = batch_size

		output = super().forward(
			betas=betas,
			global_orient=global_orient,
			body_pose=body_pose,
			left_hand_pose=left_hand_pose,
			right_hand_pose=right_hand_pose,
			transl=transl,
			expression=expression,
			jaw_pose=jaw_pose,
			leye_pose=leye_pose,
			reye_pose=reye_pose
		)

		# return to normal...
		self.batch_size = batch_size_setting
		
		return output
	

################################################################################
## OTHER PROCESSING DETAILS
################################################################################

def get_img_processing_scheme(image_encoder_name):
	if "smplerx" in image_encoder_name:
		img_processing_scheme = "smplerx"
	else:
		img_processing_scheme = None
	return img_processing_scheme