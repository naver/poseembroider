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
import roma
import itertools

from poseembroider.datasets.base_dataset import TriModalDatasetFix
import poseembroider.config as config
import poseembroider.utils as utils


# DATASET: PoseFix
################################################################################

class PoseFix(TriModalDatasetFix):

    def __init__(self, version, split,
                 tokenizer_name=None, text_index='rand',
                 num_body_joints=config.NB_INPUT_JOINTS,
                 num_shape_coeffs=config.NB_SHAPE_COEFFS,
                 cache=True, item_format='pt',
                 flatten_data_in_script_mode=False,
                 pair_kind='any',
                 load_raw_texts=False):
        """
		load_raw_texts: whether to load **cached** raw texts
						(if cache=False, the raw texts will be loaded anyways,
						 if cache=True, the tokenized texts will be loaded, and
						 	the raw texts will be loaded only if load_raw_texts
							is True.)
        """
        TriModalDatasetFix.__init__(self,
							  version=version,
							  split=split,
							  tokenizer_name=tokenizer_name,
							  text_index=text_index,
							  num_body_joints=num_body_joints,
                              num_shape_coeffs=num_shape_coeffs,
							  cache=cache,
							  item_format=item_format,
							  pair_kind=pair_kind,
							  flatten_data_in_script_mode=flatten_data_in_script_mode)

        # load data
        if cache:
            # define cache file
            self.cache_file = os.path.join(config.DATA_CACHE_DIR, f"posefix_version_{self.version}_split_{split}"+"_{}.pkl")
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
            (self.tokenizer_name is None or os.path.isfile(self.cache_file.format(f"tokenizedtexts_tokenizer_{self.tokenizer_name}"))) and \
            os.path.isfile(self.cache_file.format("pairs"))
            

    def _create_cache(self):
        # create directory for caches if it does not yet exist
        cache_dir = os.path.dirname(self.cache_file)
        if not os.path.isdir(cache_dir):
            os.makedirs(cache_dir)
            print("Created directory:", cache_dir)
        # save caches
        utils.write_pickle({"poses":self.poses, "normalizing_rotation":self.normalizing_rotation}, self.cache_file.format("poses"), tell=True)
        if self.texts_raw is not ValueError:
            utils.write_pickle({"texts":self.texts_raw}, self.cache_file.format("rawtexts"), tell=True)
        utils.write_pickle({"texts_tokens":self.texts_tokens, "texts_length":self.texts_length, "tokenizer_name":self.tokenizer_name}, self.cache_file.format(f"tokenizedtexts_tokenizer_{self.tokenizer_name}"), tell=True)
        utils.write_pickle({"pair_2_dataid":self.pair_2_dataid, "sequence_info":self.sequence_info}, self.cache_file.format("pairs"), tell=True)
        print("Saved cache files.")


    def _load_cache(self, load_raw_texts=False):
        d = utils.read_pickle(self.cache_file.format("poses"))
        self.poses = d['poses']
        self.normalizing_rotation = d['normalizing_rotation']
        if self.tokenizer_name is not None:
            d = utils.read_pickle(self.cache_file.format(f"tokenizedtexts_tokenizer_{self.tokenizer_name}"))
            self.texts_tokens, self.texts_length = d['texts_tokens'], d['texts_length']
        if load_raw_texts:
            self.load_raw_texts()
        d = utils.read_pickle(self.cache_file.format("pairs"))
        self.pair_2_dataid = d['pair_2_dataid']
        self.sequence_info = d['sequence_info']
        print("Load data from cache.")


    # --------------------------------------------------------------------------
    # DATA IMPORT
    # --------------------------------------------------------------------------

    def _load_data(self, save_cache=False):

        # load pose pairs (all splits); {pairs_id: [pose_A_id, pose_B_id])
        all_pose_pairs = utils.read_json(os.path.join(config.POSEFIX_DIR, "pair_id_2_pose_ids.json"))

        # load split-specific, kind-specific pair ids
        self.sequence_info = {} # {pair_id: ('in'|'out')}
        for kind in ['in', 'out']:
            split_pair_ids = utils.read_json(os.path.join(config.POSEFIX_DIR, f"{self.split}_{kind}_sequence_pair_ids.json"))
            self.sequence_info.update({pair_id: kind for pair_id in split_pair_ids})
            print(f"Loaded {len(split_pair_ids)} {kind}-sequence pairs (to be pruned based on available annotations).")

        # store split-specific pair information
        all_pair_ids_in_split = list(self.sequence_info.keys())
        self.pair_2_dataid = {i:v for i,v in enumerate(all_pose_pairs) if i in all_pair_ids_in_split} # {pair_id: list of [pose_A_id, pose_B_id]}
        
        # load modifier files
        dataset_spec = self.version.split("-")[1]
        data_files = []
        if "H" in dataset_spec: # eg. posefix-H, posefix-HPP
            data_files.append(os.path.join(config.POSEFIX_DIR, "posefix_human_6157.json"))
        if "PP" in dataset_spec: # eg. posefix-PP, posefix-HPP
            data_files.append(os.path.join(config.POSEFIX_DIR, "posefix_paraphrases_4284.json"))

        # store available modifiers
        self.texts_raw = {pair_id:[] for pair_id in all_pair_ids_in_split} # {pair_id: list of texts}
        for triplet_file in data_files:
            annotations = utils.read_json(triplet_file)
            for data_id_str, c in annotations.items():
                try:
                    self.texts_raw[int(data_id_str)] += c if type(c) is list else [c]
                except KeyError:
                    # this annotation is not part of the split
                    pass 
            print("Loaded annotations from:", triplet_file)
     
        # remove all pairs for which ALL modifiers are empty
        self.remove_non_annotated_pairs()

        # parse pose information
        all_pose_ids_in_split = list(set(itertools.chain(*self.pair_2_dataid.values())))
        self.poses = {} # {data_id: 3D rotations for the body joints}
        self.normalizing_rotation = {} # {data_id: normalizing rotation} --> will make it possible to get the actual in-sequence rotation between poses A and poses B, without having to store them separately for each pair
        for data_id in all_pose_ids_in_split:
            normalized_pose, normalizing_rot = self._process_pose(data_id)
            self.poses[data_id] = normalized_pose
            self.normalizing_rotation[data_id] = normalizing_rot

        self.texts_tokens, self.texts_length = self.tokenize_texts(self.texts_raw)

        if save_cache:
            self._create_cache()


    def _process_pose(self, data_id):
        # load data
        path_to_smplx = os.path.join(config.POSEFIX_SMPLX_DIR, "{0:06d}.pkl".format(int(data_id)))
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
        # get the applied rotation 
        R = roma.rotvec_composition((pose[0:1,:], roma.rotvec_inverse(initial_rotation)))
        return pose, R


    # --------------------------------------------------------------------------
    # OVERRIDEN
    # --------------------------------------------------------------------------

    def load_raw_texts(self):
        # (specifically needed for text generation evaluation)
        if type(self.texts_raw) is not dict:
            self.texts_raw = utils.read_pickle(self.cache_file.format("rawtexts"))['texts']


    def get_pose(self, data_id, pair_id):
        p = self.poses[data_id][:self.num_body_joints].clone() # shape (nb of joints, 3)
        # if we are dealing with a pose B from an in-sequence pair, adapt its
        # global rotation to reflect the change of rotation between pose A and
        # pose B; ie. apply the same normalization as on pose A
        if self.sequence_info[pair_id] == "in" and \
            self.pair_2_dataid[pair_id][1]==data_id: # this is pose B
            data_id_A = self.pair_2_dataid[pair_id][0]
            R = roma.rotvec_composition((self.normalizing_rotation[data_id_A],
                                         roma.rotvec_inverse(self.normalizing_rotation[data_id])))
            p[0:1,:] = roma.rotvec_composition((R, p[0:1,:].clone()))
        return p


    def __getitem__(self, index, cidx=None):
        pair_id = self.index_2_id_list[index]
        data_id_A, data_id_B = self.pair_2_dataid[pair_id]
        ret = dict(data_ids_A=data_id_A, data_ids_B=data_id_B,
                     pair_ids=pair_id, indices=index, dataset=self.get_effective_version())
        if 't' in self.item_format:
            text_tokens, text_length = self.get_text(pair_id, cidx)
            ret.update(dict(texts_tokens=text_tokens, texts_lengths=text_length))
        if 'p' in self.item_format:
            pose_A = self.get_pose(data_id_A, pair_id)
            pose_B = self.get_pose(data_id_B, pair_id)
            ret.update(dict(poses_A=pose_A, poses_B=pose_B))
        return ret


# MAIN
################################################################################

if __name__ == '__main__':
    
    for split in ['train', 'val', 'test']:
        dataset = PoseFix(version="posefix-H", split=split,
                            tokenizer_name="distilbertUncased", cache=True,
                            load_raw_texts=True, item_format='pt')
