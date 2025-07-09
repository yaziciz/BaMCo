from collections import OrderedDict
from dataclasses import dataclass
import logging
import math
from typing import Tuple, Union, Callable, Optional

from copy import deepcopy
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from PIL import Image

from transformers import AutoModel,BertConfig,AutoTokenizer

from transformers import CLIPProcessor, CLIPModel

from peft import LoraConfig, get_peft_model

from .GLIMS import GLIMS

import open_clip

class NormedLinear(nn.Module):

    def __init__(self, in_features, out_features):
        super(NormedLinear, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(in_features, out_features))
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)
        self.s = 30

    def forward(self, x):
        out = F.normalize(x, dim=1).mm(F.normalize(self.weight, dim=0))
        return self.s * out

class KGEncoder(nn.Module):
    """Knowledge Encoder model (Figure 2) is a composition of multiple components:
    - A vision tower (e.g., ViT)
    - A knowledge encoder (e.g., BiomedCLIP)
    - A GLIMS module for intra-class image encoding
    - A head for the classification task
    - A normed linear layer for the contrastive learning task"""

    def __init__(self,
                num_classes: int,
                embed_dim: int = 512,
                include_images: bool = False):
        super().__init__()

        #BiomedCLIP model (Reference Image)
        self.model, _, _ = open_clip.create_model_and_transforms('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')

        #freeze the model parameters, both for vision and text backbones
        for param in self.model.visual.parameters():
            param.requires_grad = False

        for param in self.model.text.parameters():
            param.requires_grad = False

        self.embed_dim = embed_dim
        self.include_images = include_images
        self.num_classes = num_classes
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.init_parameters()

        #GLIMS module for intra-class image encoding
        self.GLIMS = GLIMS(
            img_size=(84, 84, 36),
            in_channels=3,
            feature_size=12,
            out_channels=embed_dim,
        )

        self.head = nn.Sequential(nn.Linear(self.embed_dim, self.embed_dim), nn.LayerNorm(self.embed_dim), nn.ReLU(inplace=True),
                                      nn.Linear(self.embed_dim, self.embed_dim))
        self.head_fc = nn.Sequential(nn.Linear(self.embed_dim * 2, self.embed_dim), nn.LayerNorm(self.embed_dim), nn.ReLU(inplace=True),
                                   nn.Linear(self.embed_dim, self.embed_dim))
        self.fc = NormedLinear(self.embed_dim * 2, self.num_classes)
    
    def init_parameters(self):
        nn.init.constant_(self.logit_scale, np.log(1 / 0.07))
    
    def encode_image(self, image):
        image_features = self.model.encode_image(image)
        image_features = F.normalize(image_features, dim=1)
        return image_features

    def encode_text(self, text):
        text_features = self.model.encode_text(text)
        text_features = F.normalize(self.head(text_features), dim=1)
        return text_features
    
    def encode_intra_class_images(self, intra_class_images):
        intra_class_image_features = self.GLIMS(intra_class_images)
        intra_class_image_features = F.normalize(intra_class_image_features, dim=1)
        return intra_class_image_features
    
    def forward(self, images_all, text):

        image = images_all[0]
        intra_class_images = images_all[1]

        if(self.include_images):
            image_features = self.encode_image(image)

        text_features = self.encode_text(text)

        intra_class_image_features = self.encode_intra_class_images(intra_class_images)

        #use cls_logits for the cross-entropy task
        cls_logits = self.fc(torch.cat([image_features, intra_class_image_features], dim=1))
        centers_logits = F.normalize(self.head_fc(self.fc.weight.T), dim=1)

        if(self.include_images):
            return image_features, text_features, intra_class_image_features, cls_logits, centers_logits, self.logit_scale.exp()
        else:
            return text_features, self.logit_scale.exp()


    
