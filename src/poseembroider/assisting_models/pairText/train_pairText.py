##############################################################
## text2pose and poseembroider                              ##
## Copyright (c) 2022, 2023, 2024                           ##
## Institut de Robotica i Informatica Industrial, CSIC-UPC  ##
## and Naver Corporation                                    ##
## Licensed under the CC BY-NC-SA 4.0 license.              ##
## See project root for license details.                    ##
##############################################################

import torch
from tqdm import tqdm
import math
import sys
import os
os.umask(0x0002)
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

from text2pose.encoders.tokenizers import Tokenizer, get_tokenizer_name
from text2pose.retrieval_modifier.model_retrieval_modifier import PairText
import text2pose.utils_logging as logging

import poseembroider.evaluator as evaluator
from poseembroider.option import get_args_parser
from poseembroider.trainer import GenericTrainer
from poseembroider.loss import BBC, symBBC
from poseembroider.datasets.bedlam_fix import BEDLAMFix
from poseembroider.datasets.posefix import PoseFix
from poseembroider.augmentations import DataProcessingModule


class PairTextTrainer(GenericTrainer):

    def __init__(self, args):
        super(PairTextTrainer, self).__init__(args, retrieval_trainer=True)

    
    def load_sub_dataset(self, dataset_version, split, caption_index, tokenizer_name=None):
        
        if tokenizer_name is None: tokenizer_name = get_tokenizer_name(self.args.text_encoder_name)

        if 'posefix' in dataset_version:
            d = PoseFix(version=dataset_version,
                          split=split,
                          tokenizer_name=tokenizer_name,
                          text_index=caption_index,
                          num_body_joints=self.args.num_body_joints,
                          item_format='pt')
        elif 'bedlamfix' in dataset_version:
            fs_size = {"val":10000, "train":50000}[split]
            d = BEDLAMFix(version=dataset_version,
                          split=split,
                          tokenizer_name=tokenizer_name,
                          text_index=caption_index,
                          num_body_joints=self.args.num_body_joints,
                          reduced_set=fs_size,
                          item_format='pt')
        else:
            raise NotImplementedError
    
        data_size = self.args.data_size if split=="train" else None
        if data_size:
            initial_size = len(d)
            d.index_2_id_list = d.index_2_id_list[:data_size]
            print(f"[Using reduced dataset!] Size: {initial_size} --> {len(d)}")

        return d


    def init_model(self):
        print('Load model')
        self.model = PairText(text_encoder_name=self.args.text_encoder_name,
                            transformer_topping=self.args.transformer_topping,
                            latentD=self.args.latentD,
                            num_body_joints=self.args.num_body_joints)
        self.model.to(self.device)


    def get_param_groups(self):
        param_groups = []
        param_groups.append({'params': self.model.pose_encoder.parameters(), 'lr': self.args.lr*self.args.lrposemul})
        param_groups.append({'params': self.model.pose_mlp.parameters(), 'lr': self.args.lr*self.args.lrposemul})
        param_groups.append({'params': [p for k,p in self.model.text_encoder.named_parameters() if 'pretrained_text_encoder.' not in k], 'lr': self.args.lr*self.args.lrtextmul})
        param_groups.append({'params': [self.model.loss_weight]})
        return param_groups


    def init_optimizer(self):
        assert self.args.optimizer=='Adam'
        param_groups = self.get_param_groups()
        self.optimizer = torch.optim.Adam(param_groups, lr=self.args.lr)


    def init_lr_scheduler(self):
        self.lr_scheduler = None
        if self.args.lr_scheduler == "stepLR":
            self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,
                                                    step_size=self.args.lr_step,
                                                    gamma=self.args.lr_gamma,
                                                    last_epoch=-1)


    def init_other_training_elements(self):
        # data processing & augmentations
        tokenizer_name = get_tokenizer_name(self.args.text_encoder_name)
        tokenizer = Tokenizer(tokenizer_name)
        self.data_processing_module_train = DataProcessingModule(
                                            phase="train",
                                            nb_joints=self.args.num_body_joints,
                                            lr_flip_proba=0.5 if self.args.apply_LR_augmentation else 0,
                                            tokenizer=tokenizer,
                                            )
        self.data_processing_module_val = DataProcessingModule(
                                            phase="eval",
                                            nb_joints=self.args.num_body_joints,
                                            )
        

    def training_epoch(self, epoch):
        train_stats = self.one_epoch(epoch=epoch, is_training=True)
        return train_stats
    

    def validation_epoch(self, epoch):
        val_stats = {}
        if self.args.val_every and (epoch+1)%self.args.val_every==0:
            val_stats = self.validate(epoch=epoch)
        return val_stats
    

    def one_epoch(self, epoch, is_training):

        self.model.train(is_training)

        # define loggers
        metric_logger = logging.MetricLogger(delimiter="  ")
        if is_training:
            prefix, sstr = '', 'train'
            metric_logger.add_meter(f'{sstr}_lr', logging.SmoothedValue(window_size=1, fmt='{value:.6f}'))
        else:
            prefix, sstr = '[val] ', 'val'
        header = f'{prefix}Epoch: [{epoch}]'
        
        # define dataloader & other elements
        if is_training:
            data_loader = self.data_loader_train
        if not is_training:
            data_loader = self.data_loader_val

        # iterate over the batches
        for data_iter_step, item in enumerate(metric_logger.log_every(data_loader, self.args.log_step, header)):

            # truncate within the batch, based on the longest text
            item["texts_tokens"] = item["texts_tokens"][:,:max(item["texts_lengths"])]

            # augmentations
            if is_training:
                item = self.data_processing_module_train(item)
            else:
                item = self.data_processing_module_val(item)

            # forward; compute scores
            with torch.set_grad_enabled(is_training):
                poses_features, texts_features = self.model(
                                                        poses_A=item['poses_A'].to(self.device),
                                                        captions=item['texts_tokens'].to(self.device),
                                                        caption_lengths=item['texts_lengths'].to(self.device),
                                                        poses_B=item['poses_B'].to(self.device),
                                                        )
                score_t2p = texts_features.mm(poses_features.t()) * self.model.loss_weight
            
            # compute loss
            if self.args.retrieval_loss == "BBC":
                loss = BBC(score_t2p)
            elif self.args.retrieval_loss == "symBBC":
                loss = symBBC(score_t2p)
            else:
                raise NotImplementedError

            loss_value = loss.item()
            if not math.isfinite(loss_value):
                print("Loss is {}, stopping training".format(loss_value))
                sys.exit(1)

            # training step
            if is_training:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            # format data for logging
            scalars = [('loss', loss_value)]
            if is_training:
                lr_value = self.optimizer.param_groups[0]["lr"]
                scalars += [('lr', lr_value)]

            # actually log
            self.add_data_to_log_writer(epoch, sstr, scalars=scalars, is_training=is_training, data_iter_step=data_iter_step, total_steps=len(data_loader))
            self.add_data_to_metric_logger(metric_logger, sstr, scalars)

        print("Averaged stats:", metric_logger)
        return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


    def validate(self, epoch):

        self.model.eval()

        recalls, loss_value = evaluator.compute_eval_metrics_p2t_t2p(model=self.model,
                                                                     dataset=self.data_loader_val.dataset,
                                                                     device=self.device,
                                                                     infer_features_func=infer_features,
                                                                     compute_loss=True,
                                                                     loss_func=eval(self.args.retrieval_loss))
        val_stats = {"loss": loss_value}
        val_stats.update(recalls)

        # log
        self.add_data_to_log_writer(epoch, 'val', scalars=[('loss', loss_value), ('validation', recalls)], should_log_data=True)
        print(f"[val] Epoch: [{epoch}] Stats: " + "  ".join(f"{k}: {round(v, 3)}" for k,v in val_stats.items()) )
        return val_stats


def infer_features(model, dataset, device):
	batch_size = 32
	data_loader = torch.utils.data.DataLoader(
		dataset, sampler=None, shuffle=False,
		batch_size=batch_size,
		num_workers=8,
		pin_memory=True,
		drop_last=False
	)

	poses_features = torch.zeros(len(dataset), model.latentD).to(device)
	texts_features = torch.zeros(len(dataset), model.latentD).to(device)

	for i, batch in tqdm(enumerate(data_loader)):
		poses_A = batch['poses_A'].to(device)
		poses_B = batch['poses_B'].to(device)
		caption_tokens = batch['texts_tokens'].to(device)
		caption_lengths = batch['texts_lengths'].to(device)
		caption_tokens = caption_tokens[:,:caption_lengths.max()]
		with torch.inference_mode():
			pfeat, tfeat = model(poses_A, caption_tokens, caption_lengths, poses_B)
			poses_features[i*batch_size:i*batch_size+len(poses_A)] = pfeat
			texts_features[i*batch_size:i*batch_size+len(poses_A)] = tfeat

	return poses_features, texts_features


if __name__ == '__main__':
    
    argparser = get_args_parser()
    args = argparser.parse_args()
    
    PairTextTrainer(args)()