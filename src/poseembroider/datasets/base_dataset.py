##############################################################
## poseembroider                                            ##
## Copyright (c) 2024                                       ##
## Institut de Robotica i Informatica Industrial, CSIC-UPC  ##
## and Naver Corporation                                    ##
## Licensed under the CC BY-NC-SA 4.0 license.              ##
## See project root for license details.                    ##
##############################################################

import random
import torch
import roma
from tqdm import tqdm
import smplx
import torchvision.transforms as transforms
import traceback

from text2pose.encoders.tokenizers import Tokenizer

import poseembroider.config as config

# Dataset skeletons:
# * TriModalDataset
# * TriModalDatasetScript
# * TriModalDatasetFix

# Processing processes:
# * PoseNormalizer
# * PaddResize_smplerx
# * padd_and_resize_keeping_aspect_ratio
# * get_scaled_bbox


# TRI-MODAL DATASET (general class)
################################################################################

class TriModalDataset():
    def __init__(self, version, split,
                  tokenizer_name=None, text_index='rand',
                  img_processing_scheme="smplerx",
                  num_body_joints=config.NB_INPUT_JOINTS,
                  num_shape_coeffs=config.NB_SHAPE_COEFFS,
                  cache=True, item_format='ipt'):
        """
        version: dataset version
        split: data split
        tokenizer_name: name of the text tokenizer, if any should be used
        text_index: method to select a text sample when several are available
        img_processing_scheme: name of the method to pre-process images
        num_body_joints: number of SMPL-X body joints to consider
        num_shape_coeffs: number of SMPL-X body shape coefficients to consider 
        cache: whether to create a cache
        item_format: (ipt|ip|pt|it|p) ... tells what modalities to load, where:
                     i:image, p:pose, t:text
                     (letters must appear in alphabetic order)
        """

        # dataset specificities
        self.version = version
        self.split = split
        self.tokenizer_name = tokenizer_name
        self.img_processing_scheme = img_processing_scheme

        # data format
        assert type(text_index) is int or text_index in ['deterministic-mix', 'rand', 'all']
        self.text_index = text_index
        self.num_body_joints = num_body_joints
        self.num_shape_coeffs = num_shape_coeffs
        
        # item format: redirect to the suitable __getitem__ method
        self.item_format = item_format
        
        # data location
        self.cache = cache
        self.image_dir = ValueError

        # init data holders
        self.images = ValueError
        self.camera_transf = ValueError # gives the transformation to get the pose in the image from the normalized pose
        self.texts_raw, self.texts_tokens, self.texts_length = ValueError, ValueError, ValueError
        self.poses, self.shapes = ValueError, ValueError

        # processing functions
        self.basic_image_transform = padd_and_resize_keeping_aspect_ratio(img_processing_scheme)

        # NOTE: child classes must run self.setup_index_2_id_list() at the end
        # of their __init__() method, to create the link between continuous
        # ordered indices (used to call __getitem__) and data IDs (discontinuous
        # set: some numbers could be missing); and further define the actual
        # size of the dataset.


    # --------------------------------------------------------------------------
    # BASICS
    # --------------------------------------------------------------------------

    def __len__(self):
        return len(self.index_2_id_list)
        # NOTE: may comment here and return a small number for debug 


    def get_effective_version(self):
        return self.version
    

    def get_stats(self):
        print(f"[{self.split}][{self.version}] Dataset loaded:")
        # -- images
        if type(self.images) is dict:
            print(f"  * {len(self.images)} images" + \
                (" (camera transformations available)" if type(self.camera_transf) is dict else ""))
        # -- texts
        if type(self.texts_raw) is dict:
            print(f"  * {len(self.texts_raw)} texts")
        elif type(self.texts_tokens) is dict:
            print(f"  * {len(self.texts_tokens)} texts (tokenized version, raw texts not available)")
        elif self.tokenizer_name is not None:
            print("  * (could not find proper text attribute)")
            import sys; sys.exit()
        # -- poses
        if type(self.poses) is dict:
            print(f"  * {len(self.poses)} poses"  + \
                (" (shapes available)" if type(self.shapes) is dict else ""))
            
        print(f"  * Effective size: {self.__len__()}")


    def setup_index_2_id_list(self):
        # [must be overriden in the child class]
        # Create a link between continuous ordered indices and actual data ids.
        # The process sorts the data_ids, which may NOT follow a
        # farther-sampling order (eg. BEDLAM first sorts per image then per
        # human).
        raise NotImplementedError


    def init_tokenizer(self):
        if not hasattr(self, "tokenizer") and self.tokenizer_name is not None:
            self.tokenizer = Tokenizer(self.tokenizer_name)


    def tokenize_texts(self, texts_raw):
        # tokenize & padd texts
        if self.tokenizer_name:
            self.init_tokenizer()
            texts_tokens = {data_id: [self.tokenizer(t) for t in tl] for data_id, tl in tqdm(texts_raw.items())}
            texts_length = {data_id: [len(t) for t in tl] for data_id, tl in texts_tokens.items()}
            texts_tokens = {data_id: [torch.cat( (t, self.tokenizer.pad_token_id * torch.ones( self.tokenizer.max_tokens-len(t), dtype=t.dtype) ), dim=0) \
                                    for t in tl] for data_id, tl in tqdm(texts_tokens.items())}
            return texts_tokens, texts_length
        return ValueError, ValueError
    
    
    def reduce_text_number_to_minimum_common_multiplicity(self):
        # ensures every dataset element has the same number of text samples
        mm = float('inf')
        # get minimum multiplicity
        for ind in range(self.__len__()):
            data_id = self.index_2_id_list[ind]
            n = len(self.texts_tokens[data_id])
            if n < mm: mm = n
        # reduce data
        self.texts_tokens = {k:v[:mm] for k,v in self.texts_tokens.items()}
        self.texts_length = {k:v[:mm] for k,v in self.texts_length.items()}
        return mm


    # --------------------------------------------------------------------------
    # DATA LOADING
    # --------------------------------------------------------------------------
    # assume that data is always available
    # (use `item_format` in getitem-like methods (implemented in child classes)
    # to avoid loading something that is not available)

    def load_image(self, data_id):
        # must be overriden in the child class
        raise NotImplementedError


    def load_raw_texts(self):
        # needed for evaluation with NLP metrics
        raise NotImplementedError
    

    def get_text_index(self, n, data_id):
        # define which text to consider (ie. its "index") in the list of texts
        # available for a given data_id
        if self.text_index=='deterministic-mix':
            return data_id % n
        elif self.text_index=='rand':
            return random.randint(0, n-1)
        elif self.text_index=='all':
            return None
        elif self.text_index < n:
            return self.text_index
        raise ValueError


    def get_image(self, data_id):
        # open & load (& crop) image
        image = self.load_image(data_id)
        # return padded & resized tensor
        image = self.basic_image_transform(image)
        return image


    def get_pose(self, data_id):
        p = self.poses[data_id][:self.num_body_joints] # shape (nb of joints, 3)
        return p


    def get_shape(self, data_id):
        p = self.shapes[data_id][:self.num_shape_coeffs] # shape (nb shape coeffs)
        return p


    def get_text(self, data_id, cidx=None): # NOTE: `data_id` here could be a `pair_id`
        cidx = cidx if cidx else self.get_text_index(len(self.texts_tokens[data_id]), data_id)
        if cidx is not None: # Note that cidx could be 0!
            return self.texts_tokens[data_id][cidx], \
                    self.texts_length[data_id][cidx]
        else:
            return torch.stack(self.texts_tokens[data_id]), \
                    torch.tensor(self.texts_length[data_id]) # shape (nb texts, ...)


    def get_raw_text(self, data_id, cidx=None): # NOTE: `data_id` here could be a `pair_id`
        cidx = cidx if cidx else self.get_text_index(len(self.texts_raw[data_id]), data_id)
        text_raw = self.texts_raw[data_id][cidx]
        return text_raw


    def get_all_raw_texts(self, data_id): # NOTE: `data_id` here could be a `pair_id`
        return self.texts_raw[data_id]

    
    def get_camera_rotation(self, data_id):
        raise NotImplementedError
    

    def get_all_raw_texts(self, index):
        pair_id = self.index_2_id_list[index]
        return self.texts_raw[pair_id]


    # --------------------------------------------------------------------------

    def __getitem__(self, index, cidx=None):
        raise NotImplementedError



# TRI-MODAL DATASET (Script)
################################################################################

class TriModalDatasetScript(TriModalDataset):
    def __init__(self, version, split,
                  tokenizer_name=None, text_index='rand',
                  img_processing_scheme="smplerx",
                  num_body_joints=config.NB_INPUT_JOINTS,
                  num_shape_coeffs=config.NB_SHAPE_COEFFS,
                  cache=True, item_format='ipt'):
        """
        Check TriModalDataset description.
        """
        super(TriModalDatasetScript, self).__init__(version=version,
                                                    split=split,
                                                    tokenizer_name=tokenizer_name,
                                                    text_index=text_index,
                                                    img_processing_scheme=img_processing_scheme,
                                                    num_body_joints=num_body_joints,
                                                    num_shape_coeffs=num_shape_coeffs,
                                                    cache=cache, item_format=item_format)

        # NOTE: child classes must run self.setup_index_2_id_list() at the end
        # of their __init__() method, to create the link between continuous
        # ordered indices and data ids.

    # --------------------------------------------------------------------------

    def setup_index_2_id_list(self):
        # Create a link between continuous ordered indices and actual data ids.
        # The process sorts the data_ids, which may NOT follow a
        # farther-sampling order (eg. BEDLAM first sorts per image then per
        # human).
        # --- list which data modalities are important to determine the
        # available sample set, depending on the required item format
        considered_lists = (['images'] if 'i' in self.item_format else []) + \
                            (['poses'] if 'p' in self.item_format else []) + \
                            (['texts'] if 't' in self.item_format else [])
        # --- check which data samples have an item of each required modality
        try:
            attribute_suffix = lambda modality_name: \
                '_tokens' if (modality_name=='texts' and not self.texts_tokens==ValueError) else \
                '_raw' if (modality_name=='texts' and not self.texts_raw==ValueError) else \
                ''
            extract_available_data_ids = lambda modality_name: set(eval(f"self.{modality_name}{attribute_suffix(modality_name)}", globals(), {'self': self} ).keys())
            self.index_2_id_list = list(set.intersection(*map(extract_available_data_ids, considered_lists)))
        except TypeError:
            print(traceback.format_exc())
            print(f"Check that `item_format` (current: {self.item_format}) fits the data available.")


    def __getitem__(self, index, cidx=None):
        data_id = self.index_2_id_list[index]
        ret = dict(data_ids=data_id, indices=index, dataset=self.get_effective_version())
        # --- load modalities
        if 't' in self.item_format:
            text_tokens, text_length = self.get_text(data_id, cidx)
            ret.update(dict(texts_tokens=text_tokens, texts_lengths=text_length))
        if 'i' in self.item_format:
            ret.update(dict(images=self.get_image(data_id)))
        if 'p' in self.item_format:
            ret.update(dict(poses=self.get_pose(data_id)))
        try:
            ret.update(dict(cam_rot=self.get_camera_rotation(data_id)))
        except (TypeError, NotImplementedError): pass # not available
        try:
            ret.update(dict(shapes=self.get_shape(data_id)))
        except (TypeError, NotImplementedError): pass # not available
        return ret


# TRI-MODAL DATASET (Fix)
################################################################################

class TriModalDatasetFix(TriModalDataset):
    def __init__(self, version, split,
                  tokenizer_name=None, text_index='rand',
                  img_processing_scheme="smplerx",
                  num_body_joints=config.NB_INPUT_JOINTS,
                  num_shape_coeffs=config.NB_SHAPE_COEFFS,
                  cache=True, item_format='ipt',
                  load_script=False, flatten_data_in_script_mode=False,
                  pair_kind='any'):
        """
        Check TriModalDataset description for the first arguments.

        pair_kind: ('any'|'in'|'out'), defines the types of pairs to
                    consider (ie. whether poses of the same pair should belong
                    to the same sequence ('in'), different sequences ('out'), or
                    if any kind is fine ('any').
        load_script: whether to load element descriptions ("script")
        flatten_data_in_script_mode: whether to 'cut' pairs in 2 so as to
                                     consider all data points independently (in
                                     spirit, it more or less boils down to
                                     converting the loaded TriModalDatasetFix
                                     dataset to a TriModalDatasetScript)
        """
        super(TriModalDatasetFix, self).__init__(version=version,
                                                 split=split,
                                                 tokenizer_name=tokenizer_name,
                                                 text_index=text_index,
                                                 img_processing_scheme=img_processing_scheme,
                                                 num_body_joints=num_body_joints,
                                                 num_shape_coeffs=num_shape_coeffs,
                                                 cache=cache,
                                                 item_format=item_format)

        self.pair_kind = pair_kind
        self.flatten_data_in_script_mode = flatten_data_in_script_mode # breaks data pairs in 2 single data points

        # added data holders
        self.pair_2_dataid = ValueError # gives the data ids of the elements composing the pair
        self.sequence_info = ValueError # tells wether this is an "in"-sequence or "out"-of-sequence pair
        # NOTE: expected format of things:
        # self.pair_2_dataid: {pair_id: (element_A_data_id, element_B_data_id)}
        # self.texts_*: {pair_id: ...}
        # self.poses, self.images, self.shapes: {data_id: ...}

        self.load_script = load_script
        self.texts_raw_descriptions = ValueError # {data_id: ...}
        self.texts_tokens_descriptions = ValueError # {data_id: ...}
        self.texts_length_descriptions = ValueError # {data_id: ...}

        # NOTE: child classes must run self.setup_index_2_id_list() at the end
        # of their __init__() method, to create the link between continuous
        # ordered indices and data ids.

    # --------------------------------------------------------------------------

    def get_stats(self):
        super().get_stats()
        # -- others
        if type(self.sequence_info) is dict:
            for k in sorted(set(self.sequence_info.values())):
                print(f"  * {sum([v==k for data_id, v in self.sequence_info.items() if data_id in self.index_2_id_list])} {k}-sequence pairs")
        print(f"  * {len(set.union(*map(set, self.pair_2_dataid.values())))} distinct elements used in pairs")
        if type(self.texts_raw_descriptions) is dict:
            print(f"  * {len(self.texts_raw_descriptions)} description texts")
        elif type(self.texts_tokens_descriptions) is dict:
            print(f"  * {len(self.texts_tokens_descriptions)} description texts (tokenized version, raw texts not available)")


    def setup_index_2_id_list(self):
        # Create a link between continuous ordered indices and actual data ids.
        if self.pair_kind != 'any':
            self.pair_2_dataid = {k:v for k,v in self.pair_2_dataid.items() if self.sequence_info[k] == self.pair_kind}
        try:
            pair_test = lambda l, v: v[0] in l and v[1] in l
            element_test = lambda k, v: ('i' not in self.item_format or pair_test(self.images, v)) \
                                    and ('p' not in self.item_format or pair_test(self.poses, v)) \
                                    and ('t' not in self.item_format or k in self.texts_tokens)
            self.index_2_id_list = [k for k,v in self.pair_2_dataid.items() if element_test(k,v)]
        except TypeError:
            print(traceback.format_exc())
            print(f"Check that `item_format` (current: {self.item_format}) fits the data available.")

        if self.flatten_data_in_script_mode:
            self.index_2_id_list = [self.pair_2_dataid[pair_id][0] for pair_id in self.index_2_id_list] + \
                                    [self.pair_2_dataid[pair_id][1] for pair_id in self.index_2_id_list]
            

    def remove_non_annotated_pairs(self):
        
        # -- option 1: remove all pairs for which at least one modifier is empty
        # all_txt_non_empty = lambda txt_list: sum([len(t)>0 for t in txt_list])==len(txt_list)
        # self.texts_raw = {k:v for k,v in text_data.items() if all_txt_non_empty(v)}

        # -- option 2 (chosen): remove all pairs for which ALL modifiers are empty
        initial_number_of_pairs = len(self.pair_2_dataid)
        rm_empty_texts = lambda txt_list: [t for t in txt_list if len(t)]
        removed_modifiers = 0 # keep track of numbers for information display
        all_pair_ids = list(self.pair_2_dataid.keys())
        for pair_id in tqdm(all_pair_ids):
            ctxt = rm_empty_texts(self.texts_raw[pair_id])
            rm_n = len(self.texts_raw[pair_id]) - len(ctxt)
            if len(ctxt) == 0:
                del self.texts_raw[pair_id]
                del self.pair_2_dataid[pair_id]
                del self.sequence_info[pair_id]
            elif rm_n: # removed at least one text; but not all
                self.texts_raw[pair_id] = ctxt
                removed_modifiers += rm_n
        print(f"Removed {initial_number_of_pairs - len(self.pair_2_dataid)} pairs for which all provided modifiers were empty.")
        print(f"Additionally removed {removed_modifiers} empty modifiers from some pairs which had some empty texts.")
   

    # --------------------------------------------------------------------------

    def get_text_description(self, data_id, cidx=None):
        cidx = cidx if cidx else self.get_text_index(len(self.texts_tokens_descriptions[data_id]), data_id)
        if cidx:
            return self.texts_tokens_descriptions[data_id][cidx], \
                    self.texts_length_descriptions[data_id][cidx]
        else:
            return torch.stack(self.texts_tokens_descriptions[data_id]), \
                    torch.tensor(self.texts_length_descriptions[data_id]) # shape (nb texts, ...)


    def reduce_text_number_to_minimum_common_multiplicity(self):
        # (1) apply algorithm on pair texts
        mm_tok = super().reduce_text_number_to_minimum_common_multiplicity()
        # (2) apply algorithm on item texts
        mm = float('inf')
        if self.flatten_data_in_script_mode or self.load_script:
            # get minimum multiplicity
            for ind in range(self.__len__()):
                data_id = self.index_2_id_list[ind]
                n = len(self.texts_tokens_descriptions[data_id])
                if n < mm: mm = n
            # reduce data
            self.texts_tokens_descriptions = {k:v[:mm] for k,v in self.texts_tokens_descriptions.items()}
        return (mm_tok, mm)


    def __getitem__(self, index, cidx=None):
        if self.flatten_data_in_script_mode:
            data_id = self.index_2_id_list[index]
            ret = dict(data_ids=data_id, indices=index, dataset=self.get_effective_version())
            # --- load modalities
            if 't' in self.item_format:
                text_tokens, text_length = self.get_text_description(data_id, cidx)
                ret.update(dict(texts_tokens=text_tokens, texts_lengths=text_length))
            if 'i' in self.item_format:
                ret.update(dict(images=self.get_image(data_id)))
            if 'p' in self.item_format:
                ret.update(dict(poses=self.get_pose(data_id)))
            try:
                ret.update(dict(cam_rot=self.get_camera_rotation(data_id)))
            except (TypeError, NotImplementedError): pass # not available
            try:
                ret.update(dict(shapes=self.get_shape(data_id)))
            except (TypeError, NotImplementedError): pass # not available
        else:
            pair_id = self.index_2_id_list[index]
            data_id_A, data_id_B = self.pair_2_dataid[pair_id]
            ret = dict(data_ids_A=data_id_A, data_ids_B=data_id_B,
                        pair_ids=pair_id, indices=index, dataset=self.get_effective_version())
            # --- load modalities
            if 't' in self.item_format:
                text_tokens, text_length = self.get_text(pair_id, cidx)
                ret.update(dict(texts_tokens=text_tokens, texts_lengths=text_length))
                if self.load_script:
                    texts_tokens_A, texts_lengths_A = self.get_text_description(data_id_A, cidx)
                    texts_tokens_B, texts_lengths_B = self.get_text_description(data_id_B, cidx)
                    ret.update(dict(
                        texts_tokens_A=texts_tokens_A, texts_lengths_A=texts_lengths_A,
                        texts_tokens_B=texts_tokens_B, texts_lengths_B=texts_lengths_B,
                    ))
            if 'i' in self.item_format:
                image_A = self.get_image(data_id_A)
                image_B = self.get_image(data_id_B)
                ret.update(dict(images_A=image_A, images_B=image_B))
            if 'p' in self.item_format:
                pose_A = self.get_pose(data_id_A)
                pose_B = self.get_pose(data_id_B)
                ret.update(dict(poses_A=pose_A, poses_B=pose_B))
        return ret


# PROCESSING
################################################################################

class PoseNormalizer():
    """
    Normalize a given pose so that the hips are facing front.
    """
    def __init__(self, device='cpu', flat_hand_mean=True):
        self.body_model_smplx = smplx.create(config.SMPLX_BODY_MODEL_PATH,
                'smplx',
                gender='neutral',
                num_betas=config.NB_SHAPE_COEFFS,
                use_pca=False,
                flat_hand_mean=flat_hand_mean,
                batch_size=1)
        self.body_model_smplx.eval()
        self.body_model_smplx.to(device)

    def __call__(self, pose):
        """
        pose: torch tensor, size (1, nb_joints*3)
        """
        assert pose.shape[1] == 52*3 or pose.shape[1] == 22*3

        # get direction from the right hip to the left hip
        with torch.no_grad():
            bm = self.body_model_smplx(
                global_orient=pose[:,:3],
                body_pose=pose[:,3:66],
                left_hand_pose=pose[:,66:111] if len(pose[0])>66 else None,
                right_hand_pose=pose[:,111:156] if len(pose[0])>66 else None,
            )
            j = bm.joints.detach()[0] # shape (J, 3)
            body_hips = (j[1] - j[2]).view(1,3) # shape (1,3)
            body_hips = body_hips / torch.norm(body_hips, dim=1, keepdim=True)

        # define the current 2d position of the points to align
        # and the target 2d position (~ direction) those points should have
        p2d_L, t2d_L = j[1, [0,2]].view(1,2), torch.tensor([1.0,0]).view(1,2) # components X,Z
        p2d_R, t2d_R = j[2, [0,2]].view(1,2), torch.tensor([-1.0,0]).view(1,2) # components X,Z
        points_2d = torch.cat((p2d_L, p2d_R)).view(1,len(p2d_L)+len(p2d_R),2)
        targets_2d = torch.cat((t2d_L, t2d_R)).view(1,len(p2d_L)+len(p2d_R),2)
        # compute rotation
        R2d, t2d = roma.rigid_points_registration(points_2d, targets_2d)
        # convert 2d rot mat to 3d rot mat
        R3d = torch.eye(3,3).view(1,3,3)
        R3d[0,0,0] = R2d[0,0,0]
        R3d[0,2,0] = R2d[0,1,0]
        R3d[0,0,2] = R2d[0,0,1]
        R3d[0,2,2] = R2d[0,1,1]

        # apply rotation
        initial_rotation = roma.rotvec_composition([roma.rotmat_to_rotvec(R3d), pose[:,:3]])
        pose[:,:3] = initial_rotation

        return pose


def get_scaled_bbox(corners, scale_factor=1.1, max_x=1280, max_y=720):
    """
    Given the coordinates of a tight bounding box around a person in an image 
    and the image size (coordinate limits), increase the size of the given
    bounding box to include more visual context, based on the given scale factor.
    """
    x1,y1,x2,y2 = corners # tight bbox
    # using as offset for all margins a percentage of the diagonal
    offset = (scale_factor-1) * ((x2-x1)**2 + (y2-y1)**2)**0.5 / 2
    x1 -= offset
    x2 += offset
    y1 -= offset
    y2 += offset
    # sanitize
    x1,y1,x2,y2 = int(x1), int(y1), int(x2), int(y2)
    x1, x2 = max(x1, 0), min(x2, max_x)
    y1, y2 = max(y1, 0), min(y2, max_y)
    return [x1, y1, x2, y2]


def padd_and_resize_keeping_aspect_ratio(img_processing_scheme='smplerx'):
    """
    Get the transformation pipeline that will:
    - padd the image so that it respects the required input ratio while
    preserving the aspect of the original image,
    - resize the image so that it has the required size,
    - return a torch tensor.
    """
    if img_processing_scheme == "smplerx":
        return PaddResize_smplerx()
    print("Not initializing the image processing pipeline.")
    return None


class PaddResize_smplerx():
    """
    Input: PIL image
    Output: torch tensor
    """
    def __init__(self, size=(256, 192), interpolation=transforms.InterpolationMode.BICUBIC):
        # NOTE: although they load (512, 384)-sized images, they resize them to
        # (256, 192); so let's process them so they directly have the correct
        # size (ie. perform interpolation operations only once)
        self.size = size # tuple of ints
        self.interpolation = interpolation
    
    def __call__(self, img):
        # Main differences with the initial preprocessing scheme in SMPLer-X include:
        #   - the images would not be rotated (ie. resizing operation only),
        #   - the bounding boxes will be a bit smaller (scale factor would be
        #     about ~1.1 instead of ~1.25),
        #   - the images would not be processed by OpenCV functions but by PIL &
        #     pytorch functions instead (which could lead to small
        #     discrepancies, depending on the default settings of each library
        #     (antialiasing, interpolation mode etc...)).

        transforms_list = []
        # get sizes
        # Note that the dimension ordering differs depending on the element nature...
        w, h = img.size # (input data: PIL)
        H, W = self.size # (output data: torch.tensor)
        # convert to tensor in uint8 (the dataloader wants tensors to collate data)
        transforms_list += [transforms.PILToTensor()]
        # padd to the required shape
        if W/w < H/h:
            offset = int(abs((w/W*H-h))/2)
            transforms_list += [transforms.Pad(padding=[0, offset, 0, offset])]
        else:
            offset = int(abs((h/H*W-w))/2)
            transforms_list += [transforms.Pad(padding=[offset, 0, offset, 0])]
        # resize to required size
        transforms_list += [
            transforms.Resize(size=self.size, interpolation=self.interpolation, antialias=True),
        ]
        # apply all transforms
        apply_transforms = transforms.Compose(transforms_list)
        return apply_transforms(img)
    