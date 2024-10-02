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

from poseembroider.datasets.base_dataset import TriModalDatasetScript
import poseembroider.config as config
import poseembroider.utils as utils


# DATASET: PoseScript
################################################################################

class PoseScript(TriModalDatasetScript):

    def __init__(self, version, split,
                 tokenizer_name=None, text_index='rand',
                 num_body_joints=config.NB_INPUT_JOINTS,
				 num_shape_coeffs=config.NB_SHAPE_COEFFS,
                 cache=True, item_format='pt',
                 load_raw_texts=False):
        """
		load_raw_texts: whether to load **cached** raw texts
						(if cache=False, the raw texts will be loaded anyways,
						 if cache=True, the tokenized texts will be loaded, and
						 	the raw texts will be loaded only if load_raw_texts
							is True.)
        """
        TriModalDatasetScript.__init__(self,
                                version=version,
                                split=split,
                                tokenizer_name=tokenizer_name,
                                text_index=text_index,
                                num_body_joints=num_body_joints,
                                num_shape_coeffs=num_shape_coeffs,
                                cache=cache,
                                item_format=item_format)

        # load data
        if cache:
            # define cache file
            self.cache_file = os.path.join(config.DATA_CACHE_DIR, f"posescript_version_{self.version}_split_{split}"+"_{}.pkl")
            # create cache or load data from cache
            if not self._cache_exists():
                self._load_data(save_cache=True)
            else:
                self._load_cache(load_raw_texts=load_raw_texts)
            # save cache file format
        else:
            self._load_data()

        self.setup_index_2_id_list()
        self.get_stats()
        
    # --------------------------------------------------------------------------
    # CACHE PROCESSING
    # --------------------------------------------------------------------------

    def _cache_exists(self):
        return os.path.isfile(self.cache_file.format("poses")) and \
            (self.tokenizer_name is None or os.path.isfile(self.cache_file.format(f"tokenizedtexts_tokenizer_{self.tokenizer_name}")))
            

    def _create_cache(self):
        # create directory for caches if it does not yet exist
        cache_dir = os.path.dirname(self.cache_file)
        if not os.path.isdir(cache_dir):
            os.makedirs(cache_dir)
            print("Created directory:", cache_dir)
        # save caches
        utils.write_pickle({"poses":self.poses}, self.cache_file.format("poses"), tell=True)
        if self.texts_raw is not ValueError:
            utils.write_pickle({"texts":self.texts_raw}, self.cache_file.format("rawtexts"), tell=True)
        utils.write_pickle({"texts_tokens":self.texts_tokens, "texts_length":self.texts_length, "tokenizer_name":self.tokenizer_name}, self.cache_file.format(f"tokenizedtexts_tokenizer_{self.tokenizer_name}"), tell=True)
        print("Saved cache files.")


    def _load_cache(self, load_raw_texts=False):
        d = utils.read_pickle(self.cache_file.format("poses"))
        self.poses = d['poses']
        if self.tokenizer_name is not None:
            d = utils.read_pickle(self.cache_file.format(f"tokenizedtexts_tokenizer_{self.tokenizer_name}"))
            self.texts_tokens, self.texts_length = d['texts_tokens'], d['texts_length']
        if load_raw_texts:
            self.load_raw_texts()
        print("Load data from cache.")


    # --------------------------------------------------------------------------
    # DATA IMPORT
    # --------------------------------------------------------------------------

    def _load_data(self, save_cache=False):

        # load split-specific, pose ids
        split_pose_ids = utils.read_json(os.path.join(config.POSESCRIPT_DIR, f"{self.split}_ids_100k.json"))

        # load caption files
        dataset_spec = self.version.split("-")[1]
        data_files = []
        if dataset_spec == "H2": 
            data_files.append(os.path.join(config.POSESCRIPT_DIR, "posescript_human_6293.json"))
        elif dataset_spec == "A2": 
            assert False, "Not every pose has been converted to SMPL-X!!"
            data_files.append(os.path.join(config.POSESCRIPT_DIR, "posescript_auto_100k.json"))
        else:
            raise NotImplementedError

        # store available captions
        self.texts_raw = {pose_id:[] for pose_id in split_pose_ids} # {pose_id: list of texts}
        for data_file in data_files:
            annotations = utils.read_json(data_file)
            for data_id_str, c in annotations.items():
                try:
                    self.texts_raw[int(data_id_str)] += c if type(c) is list else [c]
                except KeyError:
                    # this annotation is not part of the split
                    pass 
            print("Loaded annotations from:", data_file)
        self.texts_raw = {pose_id:texts for pose_id, texts in self.texts_raw.items() if len(texts)} # filter out unavailable data

        # parse pose information
        self.poses = {} # {data_id: 3D rotations for the body joints}
        for data_id in split_pose_ids:
            try:
                self.poses[data_id] = self._process_pose(data_id)
            except FileNotFoundError:
                # this pose was not converted to SMPL-X
                continue
        assert len(self.poses)>0, "Pose files not found. Investigate with pdb."

        self.texts_tokens, self.texts_length = self.tokenize_texts(self.texts_raw)

        if save_cache:
            self._create_cache()


    def _process_pose(self, data_id):
        # load data
        path_to_smplx = os.path.join(config.POSESCRIPT_SMPLX_DIR, "{0:06d}.pkl".format(int(data_id)))
        smplx_data = utils.read_pickle(path_to_smplx)
        # format data
        pose = torch.concat([
                smplx_data['global_orient'],
                smplx_data["body_pose"],
                smplx_data["left_hand_pose"],
                smplx_data["right_hand_pose"]
            ]).to(torch.float32).view(-1, 3).detach()
        # normalize global orient
        initial_rotation = pose[:1,:].clone()
        thetax, thetay, thetaz = utils.rotvec_to_eulerangles( initial_rotation )
        zeros = torch.zeros_like(thetaz)
        pose[0:1,:] = utils.eulerangles_to_rotvec(thetax, thetay, zeros)
        return pose


    # --------------------------------------------------------------------------
    # OVERRIDEN
    # --------------------------------------------------------------------------

    def load_raw_texts(self):
        # (specifically needed for text generation evaluation)
        if type(self.texts_raw) is not dict:
            self.texts_raw = utils.read_pickle(self.cache_file.format("rawtexts"))['texts']


    def get_pose(self, data_id):
        return self.poses[data_id][:self.num_body_joints].clone() # shape (nb of joints, 3)


# MAIN
################################################################################

if __name__ == '__main__':

    for split in ['train', 'val', 'test']:
        dataset = PoseScript(version="posescript-H2", split=split,
                            tokenizer_name="distilbertUncased", cache=True,
                            load_raw_texts=True, item_format='pt')