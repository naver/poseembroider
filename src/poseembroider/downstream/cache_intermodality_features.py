##############################################################
## poseembroider                                            ##
## Copyright (c) 2024                                       ##
## Institut de Robotica i Informatica Industrial, CSIC-UPC  ##
## and Naver Corporation                                    ##
## Licensed under the CC BY-NC-SA 4.0 license.              ##
## See project root for license details.                    ##
##############################################################

import argparse
import os
from tqdm import tqdm
import torch
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import poseembroider.config as config
import poseembroider.utils as utils
import poseembroider.evaluator as evaluator
from poseembroider.datasets.bedlam_fix import BEDLAMFix
from poseembroider.datasets.bedlam_script import BEDLAMScript
from poseembroider.datasets.posefix import PoseFix
from poseembroider.augmentations import DataProcessingModule
from poseembroider.evaluate_core import load_model


## PARSER
################################################################################

parser = argparse.ArgumentParser(description='Parameters for feature caching.')
parser.add_argument('--model_path', type=str, help='Path to the model.')
parser.add_argument('--model_shortname', type=str, help='Model shortname.')
parser.add_argument('--checkpoint', default='best', help='Checkpoint to select in case the model path is incomplete. Typical choices: best|last')
parser.add_argument('--dataset', type=str, help='Dataset.')
parser.add_argument('--split', default="train", type=str, help='Split.')
parser.add_argument('--input_types', nargs='+', type=str, default=['poses'], help="Input modalities (eg. 'images', 'poses') to be given to the model to derive the intermodality features that will be cached.")


## MAIN
################################################################################

if __name__ == '__main__':
    
    # setup
    args = parser.parse_args()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    input_types = [k[:-1] for k in args.input_types] # plural --> singular
    item_format = ''.join(sorted([i[0] for i in input_types])) # minimal input format
    
    # get model path
    assert args.model_path or args.model_shortname, "Must define which model to use with --model_path or --model_shortname"
    if args.model_shortname:
        path_to_model = utils.read_json(config.PRETRAINED_MODEL_DICT)[args.model_shortname]
        if args.model_path: # security check
            assert args.model_path == path_to_model, "Mismatch between given model path and given model shortname. Check arguments, or use only one of them."
        else:
            args.model_path = path_to_model

    # load model
    args = evaluator.get_full_model_path(args)
    nb_epoch = evaluator.get_epoch(args.model_path)
    model, tokenizer_name = load_model(args.model_path, device)
    tokenizer_name = tokenizer_name if 't' in item_format else None # no need to load the tokenizer if text is not an input modality
    img_processing_scheme = utils.get_img_processing_scheme(model.image_encoder_name)

    # initialize dataset
    precision = ""
    get_virtual_ID = lambda poseid, pairid, role: poseid
    is_a_pair_dataset = False
    if "bedlamscript" in args.dataset:
        fs_size = {"val":10000, "train":50000}[args.split]
        dataset = BEDLAMScript(version=args.dataset,
                               split=args.split,
                               tokenizer_name=tokenizer_name, text_index=0,
                               img_processing_scheme=img_processing_scheme,
                               num_body_joints=model.pose_encoder.num_body_joints,
                               reduced_set=fs_size,
                               item_format=item_format)
        precision = f"_fs{fs_size}"
    elif "bedlamfix" in args.dataset:
        fs_size = {"val":10000, "train":50000}[args.split]
        dataset = BEDLAMFix(version=args.dataset,
                            split=args.split,
                            tokenizer_name=tokenizer_name, text_index=0,
                            img_processing_scheme=img_processing_scheme,
                            num_body_joints=model.pose_encoder.num_body_joints,
                            reduced_set=fs_size,
                            item_format=item_format)
        precision = f"_fs{fs_size}"
        is_a_pair_dataset = True
    elif 'posefix' in args.dataset:
        dataset = PoseFix(version=args.dataset,
                          split=args.split,
                          tokenizer_name=tokenizer_name, text_index=0,
                          num_body_joints=model.pose_encoder.num_body_joints,
                          item_format=item_format)
        is_a_pair_dataset = True
        # Patch: in the case of PoseFix, the normalized global pose orientation
        # is not the same if the pose has the role of pose A or the role of pose
        # B, in "in-sequence" pairs. As a consequence, th pose representation
        # may also differ. So the same pose, depending on the pair it belongs
        # to, and its role, can have several representations. To account for
        # this, lets create a virtual pose ID, based on the original pose ID,
        # the pair ID and its role in it.
        get_virtual_ID = lambda poseid, pairid, role: f'{poseid}_{pairid}_{role}'
    else:
        raise NotImplementedError
    
    data_processing = DataProcessingModule(phase="eval", img_processing_scheme=img_processing_scheme) # for image processing

    # init dataloader
    batch_size = 32
    data_loader = torch.utils.data.DataLoader(
        dataset, shuffle=False,
        batch_size=batch_size,
        num_workers=8,
        pin_memory=True,
        drop_last=False
    )

    # compute features
    features = {}

    def get_feat(sub_item, input_types, model):
        if model.__class__.__name__ == "PoseEmbroider":
            x = model.get_intermodality_token(**sub_item, query_modalities=input_types)
        elif model.__class__.__name__ == "Aligner":
            x = model.get_query_features(**sub_item, query_modalities=input_types)
        return x

    with torch.no_grad() and torch.inference_mode():

        for i_batch, item in tqdm(enumerate(data_loader)):
            
            # load data
            item = data_processing(item) # process images
            input_dict = {k:v.to(device) if k not in ["indices", "dataset"] else v for k,v in item.items() } # set on device
            
            if is_a_pair_dataset:
            
                # process each pose (A, B) separately
                for key in ["A", "B"]:

                    # forward pass (batch-wise)
                    sub_item = {itn.replace(f"_{key}", ""): itv for itn, itv in input_dict.items() if key in itn}
                    x = get_feat(sub_item, input_types, model)

                    # store computed features (pose-wise)
                    for i, (data_id, pair_id) in enumerate(zip(item[f"data_ids_{key}"], item["pair_ids"])):
                        virtual_data_id = get_virtual_ID(data_id.item(), pair_id.item(), key)
                        features[virtual_data_id] = x[i].view(-1).cpu()

            else:

                # forward pass (batch-wise)
                x = get_feat(input_dict, input_types, model)

                # store computed features (pose-wise)
                for i, data_id in enumerate(item[f"data_ids"]):
                    features[data_id.item()] = x[i].view(-1).cpu()

    # save features
    filename_save = f"cached_features_{args.dataset}_e{nb_epoch}_{args.split}{precision}_input-{'-'.join(args.input_types)}.pt"
    filepath_save = os.path.join(os.path.dirname(args.model_path), filename_save)
    torch.save(features, filepath_save)
    print("Saved features at:", filepath_save)
    print(f"Filemark: 'e{nb_epoch}'")