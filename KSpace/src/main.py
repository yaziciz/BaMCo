import logging
import os
import random
from datetime import datetime

import numpy as np
import torch
from torch import optim
from torch.cuda.amp import GradScaler

# from io import BytesIO
# from petrel_client.client import Client
# from petrel_client.utils.data import DataLoader
from torch.utils.data import DataLoader

from transformers import AutoTokenizer

import wandb
 
from model import KGEncoder
from dataload import UMLS_Dataset_SLAKE, UMLS_Dataset_SLAKE_test, UMLS_Dataset_PathVQA, UMLS_Dataset_PathVQA_test, UMLS_Dataset_VQARAD, UMLS_Dataset_VQARAD_test

from distributed import is_master, init_distributed_device, world_info_from_env
from params import parse_args
from scheduler import cosine_lr
from train import train_one_epoch, evaluate_one_epoch

from randaugment import rand_augment_transform
from torchvision.transforms import transforms

import open_clip

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def random_seed(seed=42, rank=0):
    torch.manual_seed(seed + rank)
    np.random.seed(seed + rank)
    random.seed(seed + rank)

def main():
    args = parse_args()

    args.distributed = False
    args.local_rank, args.rank, args.world_size = world_info_from_env()
    args.log_path = None

    # fully initialize distributed device environment
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    device = init_distributed_device(args)

    args.wandb = 'wandb' in args.report_to or 'all' in args.report_to
    args.checkpoint_path = '/mnt/storage1/ziya/BaMCo/KSpace/src/checkpoints/BiomedCLIP_VQARAD'

    #if not exist
    if not os.path.exists(args.checkpoint_path):
        os.makedirs(args.checkpoint_path)

    logging.info(f'Running with a single process. Device {args.device}.')
    random_seed(args.seed, 0)

    normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)) 
    rgb_mean = (0.48145466, 0.4578275, 0.40821073)
    ra_params = dict(translate_const=int(224 * 0.45), img_mean=tuple([min(255, round(255 * x)) for x in rgb_mean]))

    augmentation_randncls = [
        transforms.RandomResizedCrop(224, scale=(0.08, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.0)
        ], p=1.0),
        transforms.RandomGrayscale(p=0.2),
        rand_augment_transform('rand-n{}-m{}-mstd0.5'.format(2, 10), ra_params),
        transforms.ToTensor(),
        normalize,
    ]

    #Augmentation for GLIMS
    augmentation_sim = [
        transforms.RandomResizedCrop(84, scale=(0.08, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        ], p=1.0),
        transforms.RandomGrayscale(p=0.2),
        rand_augment_transform('rand-n{}-m{}-mstd0.5'.format(2, 10), ra_params),
        transforms.ToTensor(),
        normalize
    ]

    train_transform = [transforms.Compose(augmentation_randncls), transforms.Compose(augmentation_sim)]
    
    val_augmentation_randncls = [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize
    ]

    # Augmentation for GLIMS
    val_augmentation_sim = [
        transforms.Resize(84),
        transforms.CenterCrop(84),
        transforms.ToTensor(),
        normalize
    ]

    val_transform = [transforms.Compose(val_augmentation_randncls), transforms.Compose(val_augmentation_sim)]

    # initialize datasets    
    train_dataset = UMLS_Dataset_VQARAD(args, args.knowledge_graph_train, transform=train_transform)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        sampler=None,
        drop_last=True
    )

    num_batches = len(train_dataloader)
    print('Number of batches in the training dataset: ', num_batches)

    val_dataset = UMLS_Dataset_VQARAD_test(args, args.knowledge_graph_test, transform=val_transform)
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        sampler=None,
        drop_last=False
    )
    
    #The model that jointly traines a multimodal encoder with a image and text encoders with BaMCo Loss.
    model = KGEncoder(num_classes = train_dataset.num_classes, include_images = True)
    model.to(device=device)

    #print trainable parameters count
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Number of trainable parameters: ', num_params)

    random_seed(args.seed, args.rank)

    if args.grad_checkpointing:
        model.set_grad_checkpointing()
    
    #The BERT tokenizer for the text encoder.
    tokenizer = open_clip.get_tokenizer('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')

    optimizer = None
    scaler = None
    if args.knowledge_graph_train:
        assert not args.trace, 'Cannot train with traced model'
        exclude = lambda n, p: p.ndim < 2 or "bn" in n or "ln" in n or "bias" in n or 'logit_scale' in n
        include = lambda n, p: not exclude(n, p)

        named_parameters = list(model.named_parameters())
        gain_or_bias_params = [p for n, p in named_parameters if exclude(n, p) and p.requires_grad]
        rest_params = [p for n, p in named_parameters if include(n, p) and p.requires_grad]

        optimizer = optim.AdamW(
            [
                {"params": gain_or_bias_params, "weight_decay": 0.},
                {"params": rest_params, "weight_decay": args.wd},
            ],
            lr=args.lr,
            betas=(args.beta1, args.beta2),
            eps=args.eps,
        )

        scaler = GradScaler() if args.precision == "amp" else None

    # optionally resume from a checkpoint
    start_epoch = 0
    if args.resume is not None:
        if os.path.isfile(args.resume):
            checkpoint = torch.load(args.resume, map_location=device)
            if 'epoch' in checkpoint:
                start_epoch = checkpoint["epoch"]
                sd = checkpoint["state_dict"]
                if not args.distributed and next(iter(sd.items()))[0].startswith('module'):
                    sd = {k[len('module.'):]: v for k, v in sd.items()}
                model.load_state_dict(sd)
                if optimizer is not None:
                    optimizer.load_state_dict(checkpoint["optimizer"])
                if scaler is not None and 'scaler' in checkpoint:
                    scaler.load_state_dict(checkpoint['scaler'])
                logging.info(f"=> resuming checkpoint '{args.resume}' (epoch {start_epoch})")
            else:
                # loading a bare (model only) checkpoint for fine-tune or evaluation
                model.load_state_dict(checkpoint)
                logging.info(f"=> loaded checkpoint '{args.resume}' (epoch {start_epoch})")
        else:
            logging.info("=> no checkpoint found at '{}'".format(args.resume))
    
    total_steps = num_batches * args.epochs
    scheduler = cosine_lr(optimizer, args.lr, args.warmup, total_steps)
    
    loss_check = float('inf')
    val_acc_check = 0

    for epoch in range(start_epoch, args.epochs):

        loss_avg = train_one_epoch(model, tokenizer, train_dataloader, len(train_dataloader), epoch, optimizer, scaler, scheduler, args)
        val_acc = evaluate_one_epoch(model, tokenizer, val_dataloader, epoch, args)
        completed_epoch = epoch + 1
        
        # Saving checkpoints.
        checkpoint_dict = {
            "epoch": completed_epoch,
            "name": args.name,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        if scaler is not None:
            checkpoint_dict["scaler"] = scaler.state_dict()

        if args.save_most_recent:
            torch.save(checkpoint_dict, os.path.join(args.checkpoint_path,  f"epoch_latest.pt"))

        if np.abs(loss_check) > np.abs(loss_avg):
            loss_check = loss_avg
            torch.save(checkpoint_dict, os.path.join(args.checkpoint_path, f"lowest_loss.pt"))
            print('Lowest loss saved - ', loss_avg)

        if np.abs(val_acc_check) < np.abs(val_acc):
            val_acc_check = val_acc
            torch.save(checkpoint_dict, os.path.join(args.checkpoint_path, f"highest_val_acc.pt"))
            print('Highest val acc saved - ', val_acc)
        
    if args.wandb and is_master(args):
        wandb.finish()

def copy_codebase(args):
    from shutil import copytree, ignore_patterns
    new_code_path = os.path.join(args.output_dir,args.logs, args.name, "code")
    if os.path.exists(new_code_path):
        print(
            f"Error. Experiment already exists at {new_code_path}. Use --name to specify a new experiment."
        )
        return -1
    print(f"Copying codebase to {new_code_path}")
    current_code_path = os.path.realpath(__file__)
    for _ in range(3):
        current_code_path = os.path.dirname(current_code_path)
    copytree(current_code_path, new_code_path, ignore=ignore_patterns('log', 'logs', 'wandb'))
    print("Done copying code.")
    return 1


if __name__ == "__main__":
    main()
