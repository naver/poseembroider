##############################################################
## poseembroider                                            ##
## Copyright (c) 2024                                       ##
## Institut de Robotica i Informatica Industrial, CSIC-UPC  ##
## and Naver Corporation                                    ##
## Licensed under the CC BY-NC-SA 4.0 license.              ##
## See project root for license details.                    ##
##############################################################

import os
from typing import Union
import torch
import torch.nn as nn
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

import poseembroider.config as config
os.environ['TORCH_HOME'] = config.TORCH_CACHE_DIR

from poseembroider.evaluate_core import load_model as load_representation_model


class RepresentationWrapper(nn.Module):
    """
    Module that yields a representation for the input, whatever its nature
    (3D pose, image, text...), using the representation_model or cached features.
    """

    def __init__(self,
              num_body_joints=config.NB_INPUT_JOINTS,
              latentD=512,
              path_to_pretrained="",
              cached_embeddings_file:Union[bool, dict]=False, # relates to the proper cached feature file(s); if not False, it means we should use cached features
              ):
        super().__init__()
        
        self.latentD = latentD
        self.num_body_joints = num_body_joints
            
        assert path_to_pretrained, "Missing path to checkpoint for the representation_model."
        
        if cached_embeddings_file:

            self.path_to_pretrained = path_to_pretrained
            self.load_cache_features(cached_embeddings_file)

            # sanity check + forward definition
            assert latentD == self.features.shape[1]
            self.forward = self.get_cache_features_representation_model
        
        else:
            # load model
            self.representation_model, _ = load_representation_model(path_to_pretrained)

            # freeze weights
            for param in self.representation_model.parameters():
                param.requires_grad = False
            print("The weights of the representation model are made not trainable.")

            # sanity check + forward definition
            assert num_body_joints == self.representation_model.pose_encoder.num_body_joints, \
                f"Mismatch between the required number of joints ({num_body_joints}) and the number of joints in the input of the pretrained model ({self.representation_model.pose_encoder.num_body_joints})."
            assert latentD == self.representation_model.latentD
            
            if self.representation_model.__class__.__name__ == "PoseEmbroider":
                self.forward = self.get_input_features_poseembroider
            elif self.representation_model.__class__.__name__ == "Aligner":
                self.forward = self.get_input_features_aligner
            else:
                raise NotImplementedError


    def load_cache_features(self, cached_embeddings_file:dict):
        """
        Args:
            cached_embeddings_file: dict formatted as {dataset_name:filename}
                                    The files in question are expected to be
                                    located in the same directory a the model
                                    used to generate them.
        """

        self.pid_2_index = {} # {dataset_name: {pid:index}}
        self.features = [] # index -> feature (torch tensor of size latentD)
        offset = 0

        # get path to feature files
        dirpath = os.path.dirname(self.path_to_pretrained) # model directory
        for dataset_name, filename in cached_embeddings_file.items(): # iterate over the datasets

            filepath = os.path.join(dirpath, filename)

            # load features & convert them to a matrix
            features = torch.load(filepath)
            pid_2_index = {k:offset+i for i,k in enumerate(features.keys())}
            features = torch.stack([f for f in features.values()])

            # store data for the corresponding dataset (ie. make it possible to
            # have cache features for several datasets a time)
            self.pid_2_index[dataset_name] = pid_2_index
            self.features.append(features)
            offset += len(features)
            print("Loaded cache features from", filepath)

        # final formatting into a unique matrix
        self.features = torch.concat(self.features)


    def get_query_modalities_from_input_types(self, input_types):
        # For instance, if:
        # input_types = ['images', 'poses', 'texts_tokens', 'texts_lengths']
        # We should return:
        # ['images', 'poses', 'texts']
        query_modalities = input_types[:] # copy
        if 'texts_tokens' in query_modalities:
            query_modalities.remove('texts_tokens')
            query_modalities.remove('texts_lengths')
            query_modalities.append('texts')
        query_modalities = [k[:-1] for k in query_modalities] # plural --> singular
        return query_modalities


    def get_input_features_poseembroider(self, item, input_types=["images"]):
        sub_item = {it: item[it] for it in input_types}
        query_modalities = self.get_query_modalities_from_input_types(input_types)
        feat = self.representation_model.get_intermodality_token(**sub_item, query_modalities=query_modalities) # intermodality token
        return feat
    

    def get_input_features_aligner(self, item, input_types=["images"]):
        sub_item = {it: item[it] for it in input_types}
        query_modalities = self.get_query_modalities_from_input_types(input_types)
        feat = self.representation_model.get_query_features(**sub_item, query_modalities=query_modalities)
        return feat


    def get_cache_features_representation_model(self, item, input_types=None):
        # assume the cached features were obtained using the required input_type

        inds = [self.pid_2_index[dname][data_id.item()]
                    for (dname, data_id) in 
                    zip(item["dataset"], item["data_ids"])]
        
        return self.features[inds] # NOTE: need to set on the device of interest
    


class PairRepresentationWrapper(RepresentationWrapper):
    """
    Extension of the RepresentationWrapper to pair-like input items.
    """

    def __init__(self,
              num_body_joints=config.NB_INPUT_JOINTS,
              latentD=512,
              path_to_pretrained="",
              cached_embeddings_file:Union[bool, dict]=False, # relates to the proper cached feature file(s); if not False, it means we should use cached features
              ):
        super().__init__(
                        num_body_joints=num_body_joints,
                        latentD=latentD,
                        path_to_pretrained=path_to_pretrained,
                        cached_embeddings_file=cached_embeddings_file
                    )


    def load_cache_features(self, cached_embeddings_file:dict):
        super().load_cache_features(cached_embeddings_file)

        # Patch: in the case of PoseFix, the normalized global pose orientation
        # is not the same if the pose has the role of pose A or the role of pose
        # B, in "in-sequence" pairs. As a consequence, its representation may
        # also differ. So the same pose, depending on the pair it belongs to,
        # and its role, can have several representations. To account for this,
        # lets create a virtual pose ID, based on the original pose ID, the pair
        # ID and its role in it.
        virtual_ID_func_regular = lambda poseid, pairid, role: poseid
        virtual_ID_func_posefix = lambda poseid, pairid, role: f'{poseid}_{pairid}_{role}'
        self.virtual_ID = {dname: virtual_ID_func_posefix if "posefix" in dname else virtual_ID_func_regular
                                     for dname in self.pid_2_index.keys()}
        

    def get_input_features_poseembroider(self, item, input_types=["poses_A", "poses_B"]):
        features = dict()
        for key in ["A", "B"]:
            item_for_key = {it.replace(f"_{key}", ""): item[it] for it in item if key in it}
            input_types_for_key = [it.replace(f"_{key}", "") for it in input_types if key in it]
            features[f'data_{key}'] = super().get_input_features_poseembroider(item_for_key, input_types_for_key)
        return features
    

    def get_input_features_aligner(self, item, input_types=["poses_A", "poses_B"]):
        features = dict()
        for key in ["A", "B"]:
            item_for_key = {it.replace(f"_{key}", ""): item[it] for it in item if key in it}
            input_types_for_key = [it.replace(f"_{key}", "") for it in input_types if key in it]
            features[f'data_{key}'] = super().get_input_features_aligner(item_for_key, input_types_for_key)
        return features


    def get_feature_index(self, dname, data_id, pair_id, role):
        return self.pid_2_index[dname][self.virtual_ID[dname](data_id, pair_id, role)]


    def get_cache_features_representation_model(self, item, input_types=None):
        # assume the cached features were obtained using the required input_type

        inds_A = [self.get_feature_index(dname, data_id.item(), pair_id.item(), 'A')
                    for (dname, data_id, pair_id) in 
                    zip(item["dataset"], item["data_ids_A"], item["pair_ids"])]
        
        inds_B = [self.get_feature_index(dname, data_id.item(), pair_id.item(), 'B')
                    for (dname, data_id, pair_id) in 
                    zip(item["dataset"], item["data_ids_B"], item["pair_ids"])]
        
        return dict(data_A=self.features[inds_A],
                    data_B=self.features[inds_B])  # NOTE: need to set on the device of interest