##############################################################
## poseembroider                                            ##
## Copyright (c) 2024                                       ##
## Institut de Robotica i Informatica Industrial, CSIC-UPC  ##
## and Naver Corporation                                    ##
## Licensed under the CC BY-NC-SA 4.0 license.              ##
## See project root for license details.                    ##
##############################################################

import os

from poseembroider.datasets.base_dataset import TriModalDatasetScript
from poseembroider.datasets.bedlam import BEDLAM
import poseembroider.config as config
import poseembroider.utils as utils


# DATASET: BEDLAM-Script (BEDLAM + automatically generated annotations)
################################################################################

class BEDLAMScript(TriModalDatasetScript, BEDLAM):

	def __init__(self, version, split,
			  	tokenizer_name=None, text_index='rand',
				img_processing_scheme="smplerx",
			  	num_body_joints=config.NB_INPUT_JOINTS,
				num_shape_coeffs=config.NB_SHAPE_COEFFS,
				cache=True, item_format='ipt',
				reduced_set=False,
				load_raw_texts=False,
				human_visibility='any'):
		"""
		reduced_set: number or False
		load_raw_texts: whether to load **cached** raw texts
						(if cache=False, the raw texts will be loaded anyways,
						 if cache=True, the tokenized texts will be loaded, and
						 	the raw texts will be loaded only if load_raw_texts
							is True.)
		human_visibility: ('any'|'full'|'partial'), defines whether humans on
						  images are fully visible, only partially visible,
						  or both
		"""
		TriModalDatasetScript.__init__(self,
								 version=version,
								 split=split,
								 tokenizer_name=tokenizer_name,
								 text_index=text_index,
								 img_processing_scheme=img_processing_scheme,
								 num_body_joints=num_body_joints,
								 num_shape_coeffs=num_shape_coeffs,
								 cache=cache,
								 item_format=item_format)
		BEDLAM.__init__(self, split=split)
		self.reduced_set = reduced_set

		# load data
		if cache:
			cache_file = os.path.join(config.DATA_CACHE_DIR, f"bedlamscript_version_{self.version.replace('bedlamscript-', '')}_split_{split}"+"_{}.pkl")
			if reduced_set:
				cache_file = f'{cache_file[:-4]}_reduced_{reduced_set}.pkl'
			self.cache_file = cache_file
			# create cache or load data from cache
			if not self._cache_exists():
				self._load_data(reduced_set=reduced_set, save_cache=True)
			else:
				self._load_cache(load_raw_texts=load_raw_texts)
		else:
			self._load_data(reduced_set=reduced_set)

		# further filter out data depending on required full/partial human visibility
		if human_visibility=='full':
			print("Keep only data with images showing fully visible humans.")
			_ = self._filter_out([data_id for (data_id, fully_visible) in self.full_human_visibility.items() if fully_visible])
		elif human_visibility=='partial':
			print("Keep only data with images showing partially visible humans.")
			_ = self._filter_out([data_id for (data_id, fully_visible) in self.full_human_visibility.items() if not fully_visible])
		
		self.setup_index_2_id_list()
		self.get_stats()


	# --------------------------------------------------------------------------
	# CACHE PROCESSING
    # --------------------------------------------------------------------------

	def _create_cache(self):
		super()._create_cache()
		print("Saved cache files.")


	def _load_cache(self, load_raw_texts=False):
		super()._load_cache(load_raw_texts)
		print("Load data from cache.")


	def _filter_out(self, keep_indices):
		"""filter out elements that are not in the selection list of indices"""
		filter_inds = super()._filter_out(keep_indices) # get BEDLAM filtering function
		# filter supplementary data dicts
		if type(self.texts_tokens) is dict:
			self.texts_tokens = filter_inds(self.texts_tokens)
			self.texts_length = filter_inds(self.texts_length)
		if self.texts_raw is not ValueError:
			self.texts_raw = filter_inds(self.texts_raw)
		return filter_inds


	# --------------------------------------------------------------------------
	# DATA IMPORT
	# --------------------------------------------------------------------------

	def _load_data(self, reduced_set=False, save_cache=False):
		"""
		Pick relevant information from the full caches to produce smaller caches.
		"""

		# load & parse general BEDLAM information
		split_, flag_reduced_set, suffix, id_2_refs = super()._load_data(reduced_set)

		# load caption data
		caption_cache_file = os.path.join(config.BEDLAM_PROCESSED_DIR, f"bedlam_{split_}_3caps{flag_reduced_set}{suffix}.pkl")
		caption_data = utils.read_pickle(caption_cache_file)
		self.texts_raw = {} # {data_id: list of texts}
		for i, data_id in enumerate(id_2_refs):
			self.texts_raw[data_id] = caption_data[i]

		self.texts_tokens, self.texts_length = self.tokenize_texts(self.texts_raw)

		if save_cache:
			self._create_cache()


	# --------------------------------------------------------------------------
	# OVERRIDEN
	# --------------------------------------------------------------------------

	def load_raw_texts(self):
		# overriding TriModalDataset.load_raw_texts
		# NOTE: load from cache
		if type(self.texts_raw) is not dict:
			self._load_raw_texts()


	def load_image(self, data_id):
		# overriding TriModalDataset.load_image
		return super().load_bedlam_image(image_path=self.images[data_id],
								   		 bbox=self.bboxes[data_id])


	def get_camera_rotation(self, data_id):
		# overriding TriModalDataset.get_camera_transf
		return super().get_bedlam_camera_rotation(cam_rot=self.camera_transf[data_id][0])


	# --------------------------------------------------------------------------
	# OTHER
	# --------------------------------------------------------------------------

	def change_order_to_farther_sampling_order(self):
		selected = self._load_farther_sampled_ids(self.reduced_set, self.split)
		self.index_2_id_list = [s for s in selected if s in self.index_2_id_list]
		print("Switching index order to farther sampling order!")


# MAIN
################################################################################

if __name__ == '__main__':
	
	# create data caches
	for split in ["val", "train"]:
		fs_size = {"val":10000, "train":50000}[split]
		dataset = BEDLAMScript(version="bedlamscript-overlap30_res224_j16_sf11", split=split,
						tokenizer_name="distilbertUncased", reduced_set=fs_size,
						cache=True, load_raw_texts=True)