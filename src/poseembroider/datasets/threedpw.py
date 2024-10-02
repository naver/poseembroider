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
import numpy as np
import roma
from PIL import Image
from tqdm import tqdm

from poseembroider.datasets.base_dataset import TriModalDatasetScript, get_scaled_bbox
import poseembroider.config as config
import poseembroider.utils as utils


# DATASET: ThreeDPW
################################################################################

class ThreeDPW(TriModalDatasetScript):

    def __init__(self, version, split,
                tokenizer_name=None, text_index='rand',
				img_processing_scheme="smplerx",
                num_body_joints=config.NB_INPUT_JOINTS,
                num_shape_coeffs=config.NB_SHAPE_COEFFS,
                cache=True, item_format='ip'):
        super(ThreeDPW, self).__init__(
            version=version,
            split=split,
            tokenizer_name=tokenizer_name,
            text_index=text_index,
            img_processing_scheme=img_processing_scheme,
            num_body_joints=num_body_joints,
            num_shape_coeffs=num_shape_coeffs,
            cache=cache,
            item_format=item_format)

        # load data
        self.image_dir = config.THREEDPW_IMG_DIR
        if cache:
            self.cache_file = os.path.join(config.DATA_CACHE_DIR, f"threedpw_version_{self.version.replace('threedpw-', '')}_split_{split}" + "_{}.pkl")
            # create cache or load data from cache
            if not self._cache_exists():
                self._load_data(save_cache=True)
            else:
                self._load_cache()
        else:
            self._load_data()

        self.setup_index_2_id_list()
        self.get_stats()


    def _convert_split_name(self, split):
        return {'train':'train', 'val':'validation', 'test':'test'}[split]

    # --------------------------------------------------------------------------
    # CACHE PROCESSING
    # --------------------------------------------------------------------------

    def _cache_exists(self):
        return os.path.isfile(self.cache_file.format("images")) and \
                os.path.isfile(self.cache_file.format("poses"))


    def _create_cache(self):
        # create directory for caches if it does not yet exist
        cache_dir = os.path.dirname(self.cache_file)
        if not os.path.isdir(cache_dir):
            os.makedirs(cache_dir)
            print("Created directory:", cache_dir)
        # save caches
        utils.write_pickle({"criteria":self.criteria, "bboxes":self.bboxes, "images":self.images, "camera_transformation":self.camera_transf}, self.cache_file.format("images"), tell=True)
        utils.write_pickle({"poses":self.poses, "shapes":self.shapes}, self.cache_file.format("poses"), tell=True)


    def _load_cache(self):
        d = utils.read_pickle(self.cache_file.format("images"))
        self.images, self.bboxes, self.camera_transf = d['images'], d['bboxes'], d["camera_transformation"]
        self.criteria = d['criteria']
        d = utils.read_pickle(self.cache_file.format("poses"))
        self.poses = d['poses']
        self.shapes = d['shapes']


    # --------------------------------------------------------------------------
    # DATA IMPORT
    # --------------------------------------------------------------------------

    def _load_data(self, save_cache=False):

        # load pre-processed data
        split_ = self._convert_split_name(self.split)
        data = utils.read_pickle(os.path.join(config.THREEDPW_DIR, f'3dpw_{split_}.pkl'))
        # `data` is organized as follow:
        # {img_path:['focal', 'princpt', 'humans']}
        # where data[img_path]['human'] is a list a dictionaries with the following keys:
        # ['pose_format', 'pose2d', 'pose3d', 'smpl_pose', 'smpl_shape',
        # 'smpl_pose2d', 'smpl_pose3d', 'smpl_gender', 'smplx_root_pose',
        # 'smplx_body_pose', 'smplx_shape', 'smplx_pose2d', 'smplx_pose3d',
        # 'smplx_gender', 'pve_smpl2smplx']
        # We resort to only a subset of these fields.
        
        # initialization
        self.criteria = {"scale_factor":1}

        self.images = {} # {data_id: image path}
        self.poses = {} # {data_id: 3D rotations for the body joints}
        self.shapes = {} # {data_id: body shape}
        self.bboxes = {} # {data_id: [x1,y1,x2,y2]}
        self.camera_transf = {} # {data_id: (R, t)} where 'R' is the rotation to apply on the body pelvis to orient it as in the image, and 't' is the projection translation
        data_id = 0
        for img_path in tqdm(data):
            for h_ind, h_data in enumerate(data[img_path]["humans"]):
                self.images[data_id] = img_path
                self.poses[data_id], cam_rot = self._process_pose(h_data)
                self.shapes[data_id] = torch.from_numpy(h_data['smplx_shape']).view(-1)
                self.bboxes[data_id] = self._compute_bbox_from_2dkpts(h_data['smplx_pose2d'])
                self.camera_transf[data_id] = (cam_rot, None)
                data_id += 1

        if save_cache:
            self._create_cache()


    def _process_pose(self, ann):
        # load pose rotations
        pose = torch.from_numpy(np.concatenate([
                ann['smplx_root_pose'],
                ann["smplx_body_pose"],
            ])).to(torch.float32).view(-1, 3)
        # convert pose into the AMASS framework
        pose[0] = roma.rotvec_composition([torch.tensor([-torch.pi/2, 0.0, 0.0]), pose[0]])
        initial_rotation = pose[0].clone()
        # normalize (similarly to AMASS poses)
        thetax, thetay, thetaz = utils.rotvec_to_eulerangles( pose[0:1] )
        zeros = torch.zeros_like(thetaz)
        pose[0:1] = utils.eulerangles_to_rotvec(thetax, thetay, zeros)
        # find the camera transformation
        cam_rot = roma.rotvec_composition((initial_rotation, roma.rotvec_inverse(pose[0])))
        return pose, cam_rot # shape (nb of joints, 3), shape (3)


    def _compute_bbox_from_2dkpts(self, kpts):
        return [kpts[:,0].min(), kpts[:,1].min(), kpts[:,0].max(), kpts[:,1].max()]


    # --------------------------------------------------------------------------
    # OVERRIDEN
    # --------------------------------------------------------------------------

    def load_image(self, data_id, scale_factor=1.3):
        image = Image.open(os.path.join(self.image_dir, self.images[data_id]))
        # convert image
        if image.mode != 'RGB':
            image = image.convert('RGB')
        # compute scaled bounding box from tight (unsanitized) bounding box 
        max_x, max_y = image.size # the picture orientation may vary!
        bbox = get_scaled_bbox(self.bboxes[data_id], scale_factor=scale_factor, max_x=max_x, max_y=max_y)
        # crop image to the bounding box of the human
        image = image.crop(bbox)
        return image


    def get_camera_rotation(self, data_id):
        return self.camera_transf[data_id][0]


# MAIN
################################################################################

if __name__ == '__main__':
    
    for split in ['train','val','test']:
        dataset = ThreeDPW(version="threedpw-sf1", split=split, cache=True)