##############################################################
## poseembroider                                            ##
## Copyright (c) 2024                                       ##
## Institut de Robotica i Informatica Industrial, CSIC-UPC  ##
## and Naver Corporation                                    ##
## Licensed under the CC BY-NC-SA 4.0 license.              ##
## See project root for license details.                    ##
##############################################################

import os
import torch

from poseembroider.datasets.base_dataset import TriModalDatasetFix
from poseembroider.datasets.bedlam import BEDLAM
import poseembroider.config as config
import poseembroider.utils as utils


# DATASET: BEDLAM+FIX (BEDLAM + automatically generated modifiers)
################################################################################

class BEDLAMFix(TriModalDatasetFix, BEDLAM):

	def __init__(self, version, split,
			  	tokenizer_name=None, text_index='rand',
				img_processing_scheme="smplerx",
			  	num_body_joints=config.NB_INPUT_JOINTS,
				num_shape_coeffs=config.NB_SHAPE_COEFFS,
				cache=True, item_format='ipt',
				load_script=False, flatten_data_in_script_mode=False,
				pair_kind='any',
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
		TriModalDatasetFix.__init__(self,
							  version=version,
							  split=split,
							  tokenizer_name=tokenizer_name,
							  text_index=text_index,
							  img_processing_scheme=img_processing_scheme,
							  num_body_joints=num_body_joints,
							  num_shape_coeffs=num_shape_coeffs,
							  cache=cache,
							  item_format=item_format,
							  pair_kind=pair_kind,
							  load_script=load_script,
							  flatten_data_in_script_mode=flatten_data_in_script_mode)
		BEDLAM.__init__(self, split=split)
		self.reduced_set = reduced_set

		# load data
		if cache:
			# define cache file
			cache_file = os.path.join(config.DATA_CACHE_DIR, f"bedlamfix_version_{self.version.replace('bedlamfix-', '')}_split_{split}"+"_{}.pkl")
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

	def _cache_exists(self):
		return super()._cache_exists() and \
			os.path.isfile(self.cache_file.format("pairs")) and \
			(not self.load_script or self.tokenizer_name is None or os.path.isfile(self.cache_file.format(f"tokenizeddescriptions_tokenizer_{self.tokenizer_name}")))


	def _create_cache(self):
		super()._create_cache()
		utils.write_pickle({"criteria":self.criteria,
					  		"pair_2_dataid":self.pair_2_dataid,
							"sequence_info":self.sequence_info},
				self.cache_file.format("pairs"), tell=True)
		if self.load_script:
			utils.write_pickle({"criteria":self.criteria, "texts_description":self.texts_raw_descriptions}, self.cache_file.format("rawdescriptions"), tell=True)
			if self.tokenizer_name is not None:
				utils.write_pickle({"criteria":self.criteria,
						   			"texts_tokens_description":self.texts_tokens_descriptions,
									"texts_length_description":self.texts_length_descriptions,
									"tokenizer_name":self.tokenizer_name},
					self.cache_file.format(f"tokenizeddescriptions_tokenizer_{self.tokenizer_name}"), tell=True)
		print("Saved cache files.")


	def _load_cache(self, load_raw_texts=False):
		super()._load_cache(load_raw_texts)
		d = utils.read_pickle(self.cache_file.format("pairs"))
		self.pair_2_dataid = d['pair_2_dataid']
		self.sequence_info = d['sequence_info']
		if self.load_script and self.tokenizer_name is not None:
			d = utils.read_pickle(self.cache_file.format(f"tokenizeddescriptions_tokenizer_{self.tokenizer_name}"))
			self.texts_tokens_descriptions = d['texts_tokens_description']
			self.texts_length_descriptions = d['texts_length_description']
			if load_raw_texts:
				d = utils.read_pickle(self.cache_file.format("rawdescriptions"))
				self.texts_raw_descriptions = d['texts_description']
		print("Load data from cache.")


	# --------------------------------------------------------------------------
	# DATA IMPORT
	# --------------------------------------------------------------------------

	def _load_data(self, reduced_set=False, save_cache=False):
		"""
		Pick relevant information from the full caches to produce smaller caches.
		"""

		# load & parse general BEDLAM information
		split_, flag_reduced_set, suffix, id_2_refs = super()._load_data(reduced_set)

		# load pair data
		modifier_cache_file = os.path.join(config.BEDLAM_PROCESSED_DIR, f"bedlam_{split_}_%s-sequence{flag_reduced_set}_3mod{suffix}.pkl")
		if "in15_out20_t05_sim0709" in self.version:
			pair_cache_file = os.path.join(config.BEDLAM_PROCESSED_DIR, f"bedlam_fix_pairs_{split_}_%s-sequence{flag_reduced_set}{suffix}.pt")
		else: raise NotImplementedError

		pair_id = 0
		self.sequence_info = {} # {pair_id: ('in'|'out')}
		self.pair_2_dataid = {} # {pair_id: list of [pose_A_id, pose_B_id]}
		self.texts_raw = {} # {pair_id: list of texts}
		for kind in ['in', 'out']:
			# load pair info
			pair_data = torch.load(pair_cache_file % kind)
			assert pair_data['local2global_pose_ids'] == list(id_2_refs.keys()), "Not the same poses!"
			# update the pair dict
			p = pair_data['pairs'][:,[1,0]] # (pose B, pose A) --> (pose A, pose B)
			p = {pair_id+i: [pair_data['local2global_pose_ids'][pi] for pi in pp] for i, pp in enumerate(p.tolist())}
			self.pair_2_dataid.update(p)
			self.sequence_info.update({pair_id+i: kind for i in range(len(p))})
			# update the text dict
			text_data = utils.read_pickle(modifier_cache_file % kind)
			text_data = {pair_id+i: tt for i, tt in text_data.items()}
			self.texts_raw.update(text_data)
			# update `paid_id` for next round & inform about data loaded
			print(f"Loaded {len(p)} {kind}-sequence pairs.")
			pair_id += len(p)

        # remove all pairs for which ALL modifiers are empty
		self.remove_non_annotated_pairs()

		# tokenize pair instructions
		self.texts_tokens, self.texts_length = self.tokenize_texts(self.texts_raw)

		# load element-wise descriptions (-script)
		if self.load_script:
			print("Loading & tokenizing element-wise descriptions...")
			caption_cache_file = os.path.join(config.BEDLAM_PROCESSED_DIR, f"bedlam_{split_}_3caps{flag_reduced_set}{suffix}.pkl")
			caption_data = utils.read_pickle(caption_cache_file)
			self.texts_raw_descriptions = {} # {data_id: list of texts}
			for i, data_id in enumerate(self.poses.keys()):
				self.texts_raw_descriptions[data_id] = caption_data[i]
			# tokenize
			self.texts_tokens_descriptions, self.texts_length_descriptions = self.tokenize_texts(self.texts_raw_descriptions)

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


# MAIN
################################################################################

if __name__ == '__main__':

	# create data caches
	for split in ["val", "train"]:
		fs_size = {"val":10000, "train":50000}[split]
		dataset = BEDLAMFix(version="bedlamfix-overlap30_res224_j16_sf11-in15_out20_t05_sim0709", split=split,
						tokenizer_name="distilbertUncased", reduced_set=fs_size,
						cache=True, load_raw_texts=True)