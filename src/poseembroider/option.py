##############################################################
## poseembroider                                            ##
## Copyright (c) 2024                                       ##
## Institut de Robotica i Informatica Industrial, CSIC-UPC  ##
## and Naver Corporation                                    ##
## Licensed under the CC BY-NC-SA 4.0 license.              ##
## See project root for license details.                    ##
##############################################################

import os
import argparse

import poseembroider.config as config


def none_or_int(value):
    if value == 'None' or value is None:
        return None
    return int(value)


def argsort_list(seq):
    return sorted(range(len(seq)), key=seq.__getitem__)


def get_args_parser():
    """
    To know which options are taken into account for the studied model/configuration (training etc.), check whether the option is used in the naming of the model, in get_output_dir() 
    """
    parser = argparse.ArgumentParser()
    

    ### data
    parser.add_argument('--dataset', default=None, type=str, help='training dataset')
    parser.add_argument('--datasets', nargs='+', default=None, type=str, help='training datasets (if specified, --dataset is ignored)')
    parser.add_argument('--dataset_weight_balance', nargs='+', default=None, type=float, help='coefficients multiplied to the size of the smaller dataset, defining the respective size of each dataset for one epoch')
    parser.add_argument('--data_size', default=None, type=none_or_int, help="(reduced) size of the training data")
    parser.add_argument('--pair_kind', default='any', choices=('in', 'out', 'any'), help='kind of pairs to consider (in-sequence, out-of-sequence or both)')
    parser.add_argument('--pair_kinds', nargs='+', default=None, type=str, help='kind of pairs to consider for each training dataset (if specified, --pair_kind is ignored); choices: in|out|any')
    parser.add_argument('--num_body_joints', default=config.NB_INPUT_JOINTS, type=int, help="Number of body joints to consider.")
    parser.add_argument('--num_shape_coeffs', default=config.NB_SHAPE_COEFFS, type=int, help="Number of shape coefficients to consider.")


    ### model architecture
    parser.add_argument('--model', default='PoseEmbroider', choices=("PoseEmbroider", "Aligner", "PoseVAE", "PoseText", "PairText", "InstructionGenerator", "HPSEstimator"), help='name of the model')
    parser.add_argument('--latentD', type=int, help='dimension of the latent space and of the main embeddings')
    parser.add_argument('--image_encoder_name', default='smplerx_vitb32', help='name of the image encoder')
    parser.add_argument('--pose_encoder_name', default='posevae', help='name of the pose encoder')
    parser.add_argument('--text_encoder_name', default='posetext', help='name of the text encoder')
    parser.add_argument('--l2normalize', action="store_true", help="normalize modality-specific embeddings")
    parser.add_argument('--encoder_projection_type', default='layerplus', type=str, help='type of learnable projection network appended to the frozen pretrained encoders')
    parser.add_argument('--external_encoder_projection_type', default='none', type=str, help='type of learnable projection network appended to the encoders, after the projection network characterized by --encoder_projection_type')
    # -- (specific to the PoseEmbroider)
    parser.add_argument('--embroider_core_type', default='transformer', choices=("transformer", "mlp"), type=str, help='kind of network at the core of the PoseEmbroider model, combining the different input modalities')
    parser.add_argument('--no_projection_heads', action="store_true", help='the PoseEmbroider model does not have reprojection heads, the intermodality is directly compared to the modality-specific global embeddings')
    # -- (specific to downstreams)
    parser.add_argument('--pretrained_representation_model', default='', type=str, help="Shortname of the pretrained representation model yielding input features for downstream neural heads.")
    parser.add_argument('--representation_model_input_types', nargs='+', type=str, help="Input modalities ('images', 'poses', 'texts') given to the representation model responsible for producing the input features of the downstream neural head.")
    # -- (specific to the HPSEstimator)
    parser.add_argument('--predict_bodyshape', action='store_true', help='Activate to equip the HPSEstimator model with a network dedicated to body shape estimation.')
    # -- (specific to InstructionGenerator)
    parser.add_argument('--text_decoder_name', default='transformer_distilbertUncased', help='name of the text decoder')
    parser.add_argument('--comparison_latentD', type=int, help='dimension of the latent space in which belongs the output of the comparison module (can be a sequence of tokens of such dimension)')
    parser.add_argument('--decoder_latentD', default=768, type=int, help='dimension of the latent space for the text decoder')
    parser.add_argument('--decoder_nhead', default=4, type=int, help='number of heads for the text decoder transformer')
    parser.add_argument('--decoder_nlayers', default=4, type=int, help='number of layers for the text decoder transformer')
    parser.add_argument('--comparison_module_mode', default='tirg', help='module to fuse the embeddings of poses A and poses B to further generate textual feedback')
    parser.add_argument('--transformer_mode', default='crossattention', help='how to inject the multimodal data into the text decoder')
    # -- (specific to PoseText/PairText)
    parser.add_argument('--transformer_topping', help='method for obtaining the sentence embedding (transformer-based text encoders)') # "avgp", "augtokens"


    ### loss
    parser.add_argument('--retrieval_loss', default='symBBC', type=str, help='contrastive loss to train retrieval-based models')
    # -- (specific to the PoseEmbroider/Aligner)
    parser.add_argument('--single_partials', action='store_true', help="consider uni-modal input subsets (with a single modality) for training")
    parser.add_argument('--dual_partials', action='store_true', help="consider bi-modal input subsets (with two modalities) for training")
    parser.add_argument('--triplet_partial', action='store_true', help="consider tri-modal input subsets (with the three modalities) for training")
    # -- (specific to the HPSEstimator)
    parser.add_argument('--smpl_losses', action='store_true', help='consider loss terms for the vertices positions and joint positions')
    # -- (specific to the PoseVAE)
    parser.add_argument('--wloss_kld', default=1.0, type=float, help='weight for KLD losses')
    parser.add_argument('--kld_epsilon', default=0.0, type=float, help='minimum value for each component of the KLD losses')
    parser.add_argument('--wloss_v2v', default=1.0, type=float, help='weight for the reconstruction loss term: vertice positions')
    parser.add_argument('--wloss_rot', default=1.0, type=float, help='weight for the reconstruction loss term: joint rotations')
    parser.add_argument('--wloss_jts', default=1.0, type=float, help='weight for the reconstruction loss term: joint positions')
    parser.add_argument('--wloss_kldnpmul', default=1.0, type=float, help='weight for KL(Np, N0), if 0, this KL is not used for training')


    ### training (optimization)
    parser.add_argument('--epochs', default=1000, type=int, help='number of epochs')
    parser.add_argument('--batch_size', default=128, type=int, help='batch size')
    parser.add_argument('--lr_scheduler', type=str, help="learning rate scheduler")
    parser.add_argument('--lr_step', default=20, type=float, help='step for the learning rate scheduler')
    parser.add_argument('--lr_gamma', default=0.5, type=float, help='gamma for the learning rate scheduler')
    parser.add_argument('--optimizer', default='Adam', type=str, help='optimizer')
    parser.add_argument('--lr', default=0.0002, type=float, help='initial learning rate')
    parser.add_argument('--lrtextmul', default=1.0, type=float, help='learning rate multiplier for the text encoder')
    parser.add_argument('--lrposemul', default=1.0, type=float, help='learning rate multiplier for the pose model')
    parser.add_argument('--wd', default=0.0001, type=float, help='weight decay')
    # training (data augmentation)
    parser.add_argument('--no_img_augmentation', action='store_true', help='the image preprocessing is the same at training time as at evaluation phase (no augmentation)')
    parser.add_argument('--apply_LR_augmentation', action='store_true', help='randomly flip "right" and "left" during training (flip poses, texts and images)')
    # training (pretraining/finetuning)
    parser.add_argument('--pretrained', default='', type=str, help="shortname for the model to be used as pretrained model (full path registered in the file pointed by config.PRETRAINED_MODEL_DICT)")
    parser.add_argument('--cached_embeddings_file', default='', type=str, help="If this field is not empty, cached features from the representation model will be loaded and given to the downstream neural heads for training, and the representation model will not be initialized (time gain). The script will look for a .pt file in the directory of the model corresponding to --pretrained_representation_model, based on dataset, split and input type information (given through other arguments), as well as the string given under this very argument, which acts as a sort of suffix for the cached feature files (for instance, it can be denoting the epoch of the checkpoint used to produce the cached features). The .pt file is expected to be a dict of size (nb_input, --latentD) with input ids as keys and features as values. Make sure that all the characteristics of the representation model used to generate this file are also given through related arguments (eg. , --latentD etc.)")


    ### validation / evaluation
    parser.add_argument('--val_every', default=1, type=int, help='run the validation phase every N epochs')
    # -- (specific to InstructionGenerator)
    parser.add_argument('--textret_model', type=str, help='Shortname of the text-to-poses retrieval model for evaluation of generated text (can be None)')
    # -- (specific to the PoseVAE)
    parser.add_argument('--fid', type=str, default='', help='shortname of the model to be used to compute the fid')
    
    
    ### utils
    parser.add_argument('--output_dir', default='', help='output directory for experiments; automatically defined if empty')
    parser.add_argument('--subdir_prefix', default='', type=str, help='intermediate directory (at the beginning of the path) to group models on disk (eg. to separate several sets of experiments)')
    parser.add_argument('--subdir_suffix', default='', type=str, help='intermediate directory (at the end of the path) to group models on disk (eg. to separate several sets of experiments)')
    parser.add_argument('--seed', default=1, type=int, help='seed for reproduceability')
    parser.add_argument('--num_workers', default=8, type=int, help='number of workers for the dataloader')
    parser.add_argument('--saving_ckpt_step', default=10000, type=int, help='number of epochs before creating a persistent checkpoint')    
    parser.add_argument('--log_step', default=20, type=int, help='number of batchs before printing and recording the logs')

    return parser


def get_output_dir(args):
    """
    Automatically create a unique reference path based on the selected options if output_dir==''.
    """

    # utils
    add_flag = lambda t, a: t if a else '' # `t` may have a symbol '_' at the beginning or at the end


    ################################
    ### --- general specifications
    ################################

    # general architecture specifications
    ARCHITECTURE_DETAILS = args.model + f"_latentD{args.latentD}" + f'_{args.num_body_joints}bodyjts'

    # general data specifications
    assert not args.datasets or not args.dataset, "Contradictory input: either use --dataset (1 dataset) or --datasets (several datasets)"
    if args.datasets:
        # sort datasets per name to ensure set unicity in model path
        d_ind_sorted = argsort_list(args.datasets)
        d_list = [args.datasets[dind] for dind in d_ind_sorted]
        pk_list = [args.pair_kinds[dind] for dind in d_ind_sorted] if args.pair_kinds is not None else ['any']*len(d_ind_sorted)
        if args.dataset_weight_balance:
            bw_list = [args.dataset_weight_balance[dind] for dind in d_ind_sorted]
        else:
            bw_list = [None]*len(d_ind_sorted)
            args.dataset_weight_balance = [1.0 for _ in range(len(args.datasets))]

        DATASET_DETAILS = '-'.join([
                            d_list[i] \
                            + add_flag(f'BW{bw_list[i]}', bw_list[0] is not None) \
                            + add_flag(f'-pairs-{pk_list[i]}Seq', pk_list[i]!='any') \
                        for i in range(len(d_list))])
    else:
        DATASET_DETAILS = args.dataset + add_flag(f'_pairs-{args.pair_kind}Seq', args.pair_kind!='any')

    DATASET_DETAILS = 'train-'+ DATASET_DETAILS + add_flag(f"_rsize{args.data_size}", args.data_size)
    
    # general data augmentation details
    DATA_AUGMENTATION_DETAILS = add_flag('_LRflip', args.apply_LR_augmentation)

    if args.cached_embeddings_file:
        assert DATA_AUGMENTATION_DETAILS == "", "Using precomputed features (--cached_embeddings_file). Data augmentation cannot be performed."
    
    DATA_AUGMENTATION_DETAILS += add_flag('_noImgAugm', args.no_img_augmentation)
    
    # general optimization specifications
    OPTIMIZATION_DETAILS = f'{args.optimizer}_lr{args.lr}'

    # general training details
    TRAINING_DETAILS = f'B{args.batch_size}_' + add_flag(f'_pretrained_{args.pretrained}', args.pretrained)


    ################################
    ### --- per-model specifications
    ################################
    
    if args.model in ["PoseText", "PairText"]:
        ARCHITECTURE_DETAILS += f'_textencoder-{args.text_encoder_name}-{args.transformer_topping}'
        LOSS_DETAILS = args.retrieval_loss
        OPTIMIZATION_DETAILS += add_flag(f'_{args.lr_scheduler}_lrstep{args.lr_step}_lrgamma{args.lr_gamma}', args.lr_scheduler=="stepLR") + \
                                add_flag(f'textmul{args.lrtextmul}posemul{args.lrposemul}', args.lrtextmul!=1.0 or args.lrposemul!=1.0)


    elif args.model == "PoseVAE":
        LOSS_DETAILS = f'wloss_kld{args.wloss_kld}_v2v{args.wloss_v2v}_rot{args.wloss_rot}_jts{args.wloss_jts}' + \
                        add_flag(f'_kldnpmul{args.wloss_kldnpmul}', args.wloss_kldnpmul) + \
                        add_flag(f'_kldeps{args.kld_epsilon}', args.kld_epsilon>0)
        OPTIMIZATION_DETAILS += f'_wd{args.wd}'


    elif args.model in ["PoseEmbroider", "Aligner"]:
        ARCHITECTURE_DETAILS += f'_textEncoder-{args.text_encoder_name}' + \
                            f'_poseEncoder-{args.pose_encoder_name}' + \
                            f'imageEncoder-{args.image_encoder_name}' + \
                            f'_encproj-{args.encoder_projection_type}' + \
                            add_flag(f'_extEncproj-{args.external_encoder_projection_type}', args.external_encoder_projection_type!='none') + \
                            add_flag('_l2normft', args.l2normalize)
        if args.model == "PoseEmbroider":
            ARCHITECTURE_DETAILS += f'_core-{args.embroider_core_type}' + \
                                    add_flag('_noprojhead', args.no_projection_heads)

        LOSS_DETAILS = args.retrieval_loss + \
                        add_flag('_singlePartials', args.single_partials) + \
                        add_flag('_dualPartials', args.dual_partials) + \
                        add_flag('_tripletPartial', args.triplet_partial)
        
        OPTIMIZATION_DETAILS += add_flag(f'_{args.lr_scheduler}_lrstep{args.lr_step}_lrgamma{args.lr_gamma}', args.lr_scheduler=="stepLR")


    elif args.model == "InstructionGenerator":
        ARCHITECTURE_DETAILS += f'represmodel-{args.pretrained_representation_model}' + \
                                f'_inputType-{"-".join(args.representation_model_input_types)}' + \
                                f'_textdecoder-{args.text_decoder_name}' + \
                                add_flag(f'_mode-{args.transformer_mode}', args.transformer_mode!="crossattention") + \
                                f'_comp-{args.comparison_module_mode}' + \
                                add_flag(f'-clatentD{args.comparison_latentD}', args.comparison_latentD!=args.latentD) + \
                                f'_dlatentD{args.decoder_latentD}' + \
                                f'_dnhead{args.decoder_nhead}' + \
                                f'_dnlayers{args.decoder_nlayers}'
        LOSS_DETAILS = 'crossentropy' # default
        OPTIMIZATION_DETAILS += f'_wd{args.wd}'


    elif args.model == "HPSEstimator":
        ARCHITECTURE_DETAILS += f'represmodel-{args.pretrained_representation_model}' + \
                                f'_inputType-{"-".join(args.representation_model_input_types)}' + \
                                add_flag(f'_{args.num_shape_coeffs}shapecoeff', args.predict_bodyshape)
        LOSS_DETAILS = 'geodesic' + add_flag("-allsmpl", args.smpl_losses)
        OPTIMIZATION_DETAILS += f'_wd{args.wd}'


    ################################
    ### --- put everything together
    ################################

    return os.path.join(config.GENERAL_EXP_OUTPUT_DIR,
                        args.subdir_prefix,
                        ARCHITECTURE_DETAILS,
                        DATASET_DETAILS,
                        LOSS_DETAILS,
                        DATA_AUGMENTATION_DETAILS,
                        OPTIMIZATION_DETAILS,
                        TRAINING_DETAILS,
                        args.subdir_suffix,
                        f'seed{args.seed}')


if __name__=="__main__":
    # return the complete model path based on the provided arguments
    argparser = get_args_parser()
    args = argparser.parse_args()
    if args.output_dir=='':
        args.output_dir = get_output_dir(args)
    print(args.output_dir)