import json
import logging
import math
import os
import cv2
import random
import time
from PIL import Image
from contextlib import suppress
from itertools import chain

import numpy as np
import torch
import torch.nn.functional as F

try:
    import wandb
except ImportError:
    wandb = None

from loss import ClipLoss, LogitAdjust, BalSCL
from distributed import is_master
from open_clip import create_model_and_transforms, get_tokenizer
from tqdm import tqdm

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def unwrap_model(model):
    if hasattr(model, 'module'):
        return model.module
    else:
        return model


def mask_tokens(inputs,tokenizer,mlm_probability=0.15,prob_replace_mask=0.8,prob_replace_rand=0.1):
    """
    Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
    """
    labels = inputs.clone()
    # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
    probability_matrix = torch.full(labels.shape, mlm_probability)
    special_tokens_mask = [
            tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
        ]
    special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)

    probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = -100  # We only compute loss on masked tokens

    # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
    indices_replaced = torch.bernoulli(torch.full(labels.shape, prob_replace_mask)).bool() & masked_indices
    inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

    # 10% of the time, we replace masked input tokens with random word
    current_prob = prob_replace_rand / (1 - prob_replace_mask)
    indices_random = torch.bernoulli(torch.full(labels.shape, current_prob)).bool() & masked_indices & ~indices_replaced
    random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long)
    inputs[indices_random] = random_words[indices_random]

    # The rest of the time (10% of the time) we keep the masked input tokens unchanged
    return inputs, labels


def train_one_epoch(model,tokenizer, dataloader, num_batches, epoch, optimizer, scaler, scheduler, args, tb_writer=None):
    device = torch.device(args.device)
    autocast = torch.cuda.amp.autocast if args.precision == 'amp' else suppress

    model.train()
    
    criterion_ce = LogitAdjust(dataloader.dataset.cls_num_list)
    criterion_scl = BalSCL(dataloader.dataset.cls_num_list, 0.07) #Balanced Supervised Contrastive Loss

    num_batches_per_epoch = num_batches

    loss_m = AverageMeter()
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    end = time.time()

    for i, batch in enumerate(dataloader):
        step = num_batches_per_epoch * epoch + i
        scheduler(step)

        text, image, image_class, intra_class_images = batch['text'], batch['image'], batch["class"], batch["intra_class_images"]
        text = tokenizer(text, context_length=args.max_length)
        text = text.to(device=device)
        image = image.to(device=device)
        image_class = image_class.to(device=device)
        intra_class_images = intra_class_images.to(device=device)
        images_all = [image, intra_class_images]

        data_time_m.update(time.time() - end)
        optimizer.zero_grad()

        with torch.amp.autocast('cuda'):
            image_features, text_features, ici_features, cls_logits, centers_logits, logit_scale = model(images_all, text)
            
            #balanced multimodal contrastive loss between image, intra-class image and text features
            scl_loss_1 = criterion_scl(centers_logits, torch.cat([image_features.unsqueeze(1), text_features.unsqueeze(1)], dim=1), image_class)
            scl_loss_2 = criterion_scl(centers_logits, torch.cat([image_features.unsqueeze(1), ici_features.unsqueeze(1)], dim=1), image_class)
            scl_loss_3 = criterion_scl(centers_logits, torch.cat([ici_features.unsqueeze(1), text_features.unsqueeze(1)], dim=1), image_class)
            ce_loss = criterion_ce(cls_logits, image_class) #cross entropy loss using the image logits

            scl_loss = (scl_loss_1 + scl_loss_2 + scl_loss_3)/3
            total_loss = 0.5 * ce_loss + 0.5 * scl_loss

        if scaler is not None:
            scaler.scale(total_loss).backward()
            if args.horovod:
                optimizer.synchronize()
                scaler.unscale_(optimizer)
                if args.norm_gradient_clip is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.norm_gradient_clip, norm_type=2.0)
                with optimizer.skip_synchronize():
                    scaler.step(optimizer)
            else:
                if args.norm_gradient_clip is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.norm_gradient_clip, norm_type=2.0)
                scaler.step(optimizer)
            scaler.update()
        else:
            total_loss.backward()
            if args.norm_gradient_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.norm_gradient_clip, norm_type=2.0)
            optimizer.step()
        
        with torch.no_grad():
            unwrap_model(model).logit_scale.clamp_(0, math.log(100))

        batch_time_m.update(time.time() - end)
        end = time.time()
        batch_count = i + 1
        if is_master(args) and batch_count % 200 == 1:
            batch_size = text.shape[0]//2
            percent_complete = 100.0 * batch_count / num_batches_per_epoch

            # NOTE loss is coarsely sampled, just master node and per log update
            loss_m.update(total_loss.item(), batch_size)
            logit_scale_scalar = logit_scale.item()

            print(
                f"Epoch: {epoch} [{batch_count}/{num_batches_per_epoch}] "
                f"({percent_complete:.2f}%) "
                f"Loss: {loss_m.val:.5g} ({loss_m.avg:.4g}) "
                f"Data (t): {data_time_m.avg:.3f} "
                f"Batch (t): {batch_time_m.avg:.3f}, {batch_size / batch_time_m.val:#g}/s "
                f"LR: {optimizer.param_groups[0]['lr']:5f} "
                f"Logit Scale: {logit_scale_scalar:.3f}", flush=True
            )

            log_data = {
                "loss": loss_m.val,
                "data_time": data_time_m.val,
                "batch_time": batch_time_m.val,
                "samples_per_scond": args.batch_size*args.world_size / batch_time_m.val,
                "scale":  logit_scale_scalar,
                "lr": optimizer.param_groups[0]["lr"]
            }

            batch_time_m.reset()
            data_time_m.reset()

    return loss_m.avg

def evaluate_one_epoch(model, tokenizer, dataloader, epoch, args):
    """
    Evaluate the model for one epoch by computing the accuracy of the model on the top-1 and top-5 predictions
    for the image relation, image head, image definition, intra-class image relation, intra-class image head,
    intra-class image definition, relation head, relation head definition, and head head definition
    """
    device = torch.device(args.device)
    autocast = torch.cuda.amp.autocast if args.precision == "amp" else suppress
    model.eval()

    context_length = 256

    # 0: image_relation, 1: image_head, 2: image_def
    # 3: ici_relation, 4: ici_head, 5: ici_head_def
    # 6: relation_head, 7: relation_head_def, 8: head_head_def
    acc_top1 = np.zeros(9, dtype=np.float32)
    acc_top5 = np.zeros(9, dtype=np.float32)
    count = 0

    for sample in tqdm(dataloader):
        relation = sample['relation']
        head = sample['head']
        head_def = sample['head_def']
        image = sample['image'].to(device)
        intra_class_image_representation = sample['intra_class_images'].to(device)

        concatenated_text = relation + head + head_def
        images_all = [image, intra_class_image_representation]
        all_text_tokens = tokenizer(concatenated_text, context_length=context_length).to(device)

        with torch.no_grad():
            image_features, text_features, ici_features, cls_logits, _, logit_scale = model(images_all, all_text_tokens)
            n_rel = len(relation)
            n_head = len(head)
            n_def = len(head_def)
            relation_features = text_features[:n_rel]
            head_features = text_features[n_rel:n_rel+n_head]
            head_def_features = text_features[n_rel+n_head:]

        # Compute logits
        logits = [
            (logit_scale * image_features @ relation_features.t()).detach().softmax(dim=-1),      # 0
            (logit_scale * image_features @ head_features.t()).detach().softmax(dim=-1),          # 1
            (logit_scale * image_features @ head_def_features.t()).detach().softmax(dim=-1),      # 2
            (logit_scale * ici_features @ relation_features.t()).detach().softmax(dim=-1),        # 3
            (logit_scale * ici_features @ head_features.t()).detach().softmax(dim=-1),            # 4
            (logit_scale * ici_features @ head_def_features.t()).detach().softmax(dim=-1),        # 5
            (logit_scale * relation_features @ head_features.t()).detach().softmax(dim=-1),       # 6
            (logit_scale * relation_features @ head_def_features.t()).detach().softmax(dim=-1),   # 7
            (logit_scale * head_features @ head_def_features.t()).detach().softmax(dim=-1),       # 8
        ]

        sorted_indices = [torch.argsort(l, dim=-1, descending=True).cpu().numpy() for l in logits]
        preds = [[sorted_indices[j][i][:5] for i in range(n_rel)] for j in range(9)]

        # Prepare label lists for each pair
        labels = [
            relation, head, head_def,
            relation, head, head_def,
            head, head_def, head_def
        ]

        # For each sample in batch
        for i in range(n_rel):
            # Top-1
            acc_top1[0] += labels[0][preds[0][i][0]] == relation[i]
            acc_top1[1] += labels[1][preds[1][i][0]] == head[i]
            acc_top1[2] += labels[2][preds[2][i][0]] == head_def[i]
            acc_top1[3] += labels[3][preds[3][i][0]] == relation[i]
            acc_top1[4] += labels[4][preds[4][i][0]] == head[i]
            acc_top1[5] += labels[5][preds[5][i][0]] == head_def[i]
            acc_top1[6] += labels[6][preds[6][i][0]] == head[i]
            acc_top1[7] += labels[7][preds[7][i][0]] == head_def[i]
            acc_top1[8] += labels[8][preds[8][i][0]] == head_def[i]
            # Top-5
            acc_top5[0] += relation[i] in [labels[0][idx] for idx in preds[0][i]]
            acc_top5[1] += head[i] in [labels[1][idx] for idx in preds[1][i]]
            acc_top5[2] += head_def[i] in [labels[2][idx] for idx in preds[2][i]]
            acc_top5[3] += relation[i] in [labels[3][idx] for idx in preds[3][i]]
            acc_top5[4] += head[i] in [labels[4][idx] for idx in preds[4][i]]
            acc_top5[5] += head_def[i] in [labels[5][idx] for idx in preds[5][i]]
            acc_top5[6] += head[i] in [labels[6][idx] for idx in preds[6][i]]
            acc_top5[7] += head_def[i] in [labels[7][idx] for idx in preds[7][i]]
            acc_top5[8] += head_def[i] in [labels[8][idx] for idx in preds[8][i]]

        count += n_rel

    acc_top1 /= count
    acc_top5 /= count

    print(
        f'Accuracy Image Relation Top1: {acc_top1[0]}, Top5: {acc_top5[0]}, '
        f'Image Head Top1: {acc_top1[1]}, Top5: {acc_top5[1]}, '
        f'Image Def Top1: {acc_top1[2]}, Top5: {acc_top5[2]}, '
        f'ICI Relation Top1: {acc_top1[3]}, Top5: {acc_top5[3]}, '
        f'ICI Head Top1: {acc_top1[4]}, Top5: {acc_top5[4]}, '
        f'ICI Head Def Top1: {acc_top1[5]}, Top5: {acc_top5[5]}, '
        f'Relation Head Top1: {acc_top1[6]}, Top5: {acc_top5[6]}, '
        f'Relation Head Def Top1: {acc_top1[7]}, Top5: {acc_top5[7]}, '
        f'Head Head Def Top1: {acc_top1[8]}, Top5: {acc_top5[8]}'
    )
    acc_avg = (acc_top1.sum() + acc_top5.sum()) / 18
    return acc_avg

