import os
import cv2
import logging
import sys
import json
import random
import pickle
import numpy as np
import pandas as pd
from PIL import Image, ImageFile

from dataclasses import dataclass
from multiprocessing import Value

# import braceexpand

import torch
import torchvision.datasets as datasets
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler, IterableDataset, get_worker_info
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoTokenizer, AutoModel

from transformers import CLIPProcessor

import open_clip

class UMLS_Dataset(Dataset):
    def __init__(self, args, knowledge_graph_path):
        #read the json file
        with open(knowledge_graph_path, 'r') as f:
            self.kg = json.load(f)

        #for all items, get entities into an array
        self.source_image_list = []
        self.source_entity_list = []
        self.source_entity_def_list = []
        self.source_entity_cuis_list = []
        self.target_entity_list = []
        self.edge_list = []

        #self.preprocess = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32").image_processor
        _, self.preprocess, _ = open_clip.create_model_and_transforms('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')

        for item in self.kg:
            if item['Relations'] == None:
                continue

            for target in item['Relations']:
                self.source_image_list.append(os.path.join(args.image_source_folder, item['image']))
                self.source_entity_list.append(item['entity'])
                self.target_entity_list.append(target['related entity'])
                self.edge_list.append(target['relation'])

                self.source_entity_cuis_list.append(item['CUI'])
                self.source_entity_def_list.append(item['def'])

        self.kg_df = pd.DataFrame(columns=['image', 'entity', 'def', 'CUI', 'relation', 'related entity'])
        self.kg_df['image'] = self.source_image_list
        self.kg_df['entity'] = self.source_entity_list
        self.kg_df['def'] = self.source_entity_def_list
        self.kg_df['CUI'] = self.source_entity_cuis_list
        self.kg_df['relation'] = self.edge_list
        self.kg_df['related entity'] = self.target_entity_list

        #self.umls_kg_info = pd.read_csv(umls_kg_file)
        self.umls_image_source_list = self.kg_df['image']
        self.umls_kg_source_list = self.kg_df['entity']
        self.umls_kg_target_list = self.kg_df['related entity']
        self.umls_kg_edge_list = self.kg_df['relation']
        self.umls_cui_source_list = self.kg_df['CUI']
        self.umls_def_source_list = self.kg_df['def']

        self.umls_cui_target_list = self.kg_df['related entity']

        self.umls_data_len = self.kg_df.shape[0]

        print('UMLS data length: ', self.umls_data_len)
    
    def __len__(self):
        return int(self.umls_data_len)
    
    def __getitem__(self, idx):
        if("pathvqa" in self.umls_image_source_list[idx]): 
            path = "/mnt/storage1/ziya/VQA/Datasets/pathvqa/KG_dataset_pathvqa/KG_Images"
            self.umls_image_source_list[idx] = os.path.join(path, self.umls_image_source_list[idx].replace("pathvqa/", ""))
        image = self.preprocess(Image.open(self.umls_image_source_list[idx]))
        text_h = self.umls_kg_source_list[idx]
        cui_h = self.umls_cui_source_list[idx]
        def_h = self.umls_def_source_list[idx]
        text_t = self.umls_kg_target_list[idx]
        #cui_t = self.umls_cui_target_list[idx]
        text_r = self.umls_kg_edge_list[idx]
        """         
        if random.random()<0.5:
            input_text = text_h + ' [SEP] ' + text_r
            pos_text =  text_t
            cui = cui_h #cui_t
        else: """
        
        if random.random() < 0.5:
            input_text = text_r + ' [SEP] ' + text_t
            pos_text =  text_h
        else:
            input_text = text_h
            pos_text = def_h

        sample = {}
        sample['input_text'] = input_text
        sample['pos_text'] = pos_text
        sample['image'] = image
        """ try: 
            if cui[0] == 'C':
                sample['cui'] = cui
            else:
                sample['cui'] = str(0)
        except:
            sample['cui'] = str(0)
         """
        return sample
    

class UMLS_Dataset_SLAKE(Dataset):
    def __init__(self, args, knowledge_graph_path, transform):
        #read the json file
        with open(knowledge_graph_path, 'r') as f:
            self.kg = json.load(f)

        #for all items, get entities into an array
        self.source_image_list = []
        self.source_entity_list = []
        self.source_entity_def_list = []
        self.target_entity_list = []
        self.edge_list = []

        self.num_intra_class_images = args.num_intra_class_images
        self.transform = transform

        #self.preprocess = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32").image_processor
        #_, self.preprocess, _ = open_clip.create_model_and_transforms('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')

        for item in self.kg:
            if(item['def_head'] == None):
                continue

            self.source_image_list.append(os.path.join(args.image_source_folder, item['image']))
            self.source_entity_list.append(item['head_entity'].lower())
            self.target_entity_list.append(item['tail_entity'].lower())
            self.edge_list.append(item['relation'].lower())
            self.source_entity_def_list.append(item['def_head'].lower())

        self.kg_df = pd.DataFrame(columns=['image', 'entity', 'def', 'relation', 'related entity'])
        self.kg_df['image'] = self.source_image_list
        self.kg_df['entity'] = self.source_entity_list
        self.kg_df['def'] = self.source_entity_def_list
        self.kg_df['relation'] = self.edge_list
        self.kg_df['related entity'] = self.target_entity_list

        #self.umls_kg_info = pd.read_csv(umls_kg_file)
        self.umls_image_source_list = self.kg_df['image']
        self.umls_kg_source_list = self.kg_df['entity']
        self.umls_kg_target_list = self.kg_df['related entity']
        self.umls_kg_edge_list = self.kg_df['relation']
        self.umls_def_source_list = self.kg_df['def']

        self.classes = self.kg_df['entity'].unique()
        self.num_classes = len(self.classes)

        self.umls_data_len = self.kg_df.shape[0]

        #self.classes = self.kg_df['entity'].unique()

        self.cls_num_list = [len(self.kg_df[self.kg_df['entity'] == cls]) for cls in self.classes]

        print('UMLS data length: ', self.umls_data_len)
    
    def __len__(self):
        return int(self.umls_data_len)
    
    def __getitem__(self, idx):
        image = self.transform[0](Image.open(self.umls_image_source_list[idx])) #single image for classification.
        text_h = self.umls_kg_source_list[idx]
        def_h = self.umls_def_source_list[idx]
        text_t = self.umls_kg_target_list[idx]
        text_r = self.umls_kg_edge_list[idx]
        idx_class = np.where(self.classes == text_h)[0][0]

        all_intra_class_images = self.kg_df[self.kg_df['entity'] == text_h]['image'].unique().tolist()

        if random.random() < 0.8:
            selected_intra_class_images = random.choices(all_intra_class_images, k=self.num_intra_class_images)
        else: selected_intra_class_images = [self.umls_image_source_list[idx]] * self.num_intra_class_images

        #read all images and store in 3, 12, 96, 96 tensor
        intra_class_images = []
        for img in selected_intra_class_images:
            img = Image.open(img)
            img = self.transform[1](img)
            intra_class_images.append(img)

        intra_class_images = torch.stack(intra_class_images)
        intra_class_images = intra_class_images.view(3, self.num_intra_class_images, intra_class_images.shape[-1], intra_class_images.shape[-1])
                
        if random.random() < 0.33:
            text = text_h
        elif random.random() < 0.66:
            text = text_r + ' ' + text_t
        else: text = def_h

        sample = {}
        sample['text'] = text
        sample['image'] = image
        sample['class'] = idx_class
        sample['intra_class_images'] = intra_class_images
        return sample
    
class UMLS_Dataset_SLAKE_test(Dataset):
    def __init__(self, args, knowledge_graph_path, transform):
        #read the json file
        with open(knowledge_graph_path, 'r') as f:
            self.kg = json.load(f)

        #for all items, get entities into an array
        self.source_image_list = []
        self.source_entity_list = []
        self.source_entity_def_list = []
        self.target_entity_list = []
        self.edge_list = []

        self.num_intra_class_images = args.num_intra_class_images
        self.transform = transform

        #self.preprocess = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32").image_processor
        #_, self.preprocess, _ = open_clip.create_model_and_transforms('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')

        for item in self.kg:
            if(item['def_head'] == None):
                continue

            self.source_image_list.append(os.path.join(args.image_source_folder, item['image']))
            self.source_entity_list.append(item['head_entity'].lower())
            self.target_entity_list.append(item['tail_entity'].lower())
            self.edge_list.append(item['relation'].lower())
            self.source_entity_def_list.append(item['def_head'].lower())

        self.kg_df = pd.DataFrame(columns=['image', 'entity', 'def', 'relation', 'related entity'])
        self.kg_df['image'] = self.source_image_list
        self.kg_df['entity'] = self.source_entity_list
        self.kg_df['def'] = self.source_entity_def_list
        self.kg_df['relation'] = self.edge_list
        self.kg_df['related entity'] = self.target_entity_list

        #self.umls_kg_info = pd.read_csv(umls_kg_file)
        self.umls_image_source_list = self.kg_df['image']
        self.umls_kg_source_list = self.kg_df['entity']
        self.umls_kg_target_list = self.kg_df['related entity']
        self.umls_kg_edge_list = self.kg_df['relation']
        self.umls_def_source_list = self.kg_df['def']

        self.classes = self.kg_df['entity'].unique()
        self.num_classes = len(self.classes)

        self.umls_data_len = self.kg_df.shape[0]

        #self.classes = self.kg_df['entity'].unique()

        self.cls_num_list = [len(self.kg_df[self.kg_df['entity'] == cls]) for cls in self.classes]

        print('UMLS data length: ', self.umls_data_len)
    
    def __len__(self):
        return int(self.umls_data_len)
    
    def __getitem__(self, idx):
        text_h = self.umls_kg_source_list[idx]
        def_h = self.umls_def_source_list[idx]
        text_t = self.umls_kg_target_list[idx]
        text_r = self.umls_kg_edge_list[idx]

        image = self.transform[0](Image.open(self.umls_image_source_list[idx]))
        intra_class_image_representation = self.transform[1](Image.open(self.umls_image_source_list[idx]))
        intra_class_image_representation = torch.stack([intra_class_image_representation for _ in range(self.num_intra_class_images)]).view(3, self.num_intra_class_images, intra_class_image_representation.shape[-1], intra_class_image_representation.shape[-1])
        relation = text_r + ' ' + text_t
        head =  text_h
        head_def = def_h

        #repeat intra_class_image_representation for num_intra_class_images times
        sample = {}
        sample['relation'] = relation
        sample['head'] = head
        sample['head_def'] = head_def
        sample['image'] = image
        sample['intra_class_images'] = intra_class_image_representation
        return sample
    

class UMLS_Dataset_PathVQA(Dataset):
    def __init__(self, args, knowledge_graph_path, transform):
        #read the json file
        with open(knowledge_graph_path, 'r') as f:
            self.kg = json.load(f)

        #for all items, get entities into an array
        self.source_image_list = []
        self.source_entity_list = []
        self.source_entity_def_list = []
        self.target_entity_list = []
        self.edge_list = []

        self.num_intra_class_images = args.num_intra_class_images
        self.transform = transform

        #self.preprocess = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32").image_processor
        #_, self.preprocess, _ = open_clip.create_model_and_transforms('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')

        for item in self.kg:
            if(item['def_head'] == None):
                continue

            self.source_image_list.append(os.path.join(args.image_source_folder, item['image']))
            self.source_entity_list.append(item['head_entity'].lower())
            self.target_entity_list.append(item['tail_entity'].lower())
            self.edge_list.append(item['relation'].lower())
            self.source_entity_def_list.append(item['def_head'].lower())

        self.kg_df = pd.DataFrame(columns=['image', 'entity', 'def', 'relation', 'related entity'])
        self.kg_df['image'] = self.source_image_list
        self.kg_df['entity'] = self.source_entity_list
        self.kg_df['def'] = self.source_entity_def_list
        self.kg_df['relation'] = self.edge_list
        self.kg_df['related entity'] = self.target_entity_list

        #self.umls_kg_info = pd.read_csv(umls_kg_file)
        self.umls_image_source_list = self.kg_df['image']
        self.umls_kg_source_list = self.kg_df['entity']
        self.umls_kg_target_list = self.kg_df['related entity']
        self.umls_kg_edge_list = self.kg_df['relation']
        self.umls_def_source_list = self.kg_df['def']

        self.classes = self.kg_df['entity'].unique()
        self.num_classes = len(self.classes)

        self.umls_data_len = self.kg_df.shape[0]

        #self.classes = self.kg_df['entity'].unique()

        self.cls_num_list = [len(self.kg_df[self.kg_df['entity'] == cls]) for cls in self.classes]

        print('UMLS data length: ', self.umls_data_len)
    
    def __len__(self):
        return int(self.umls_data_len)
    
    def __getitem__(self, idx):
        image = self.transform[0](Image.open(self.umls_image_source_list[idx]).convert("RGB")) #single image for classification.
        text_h = self.umls_kg_source_list[idx]
        def_h = self.umls_def_source_list[idx]
        text_t = self.umls_kg_target_list[idx]
        text_r = self.umls_kg_edge_list[idx]
        idx_class = np.where(self.classes == text_h)[0][0]
        
        all_intra_class_images = self.kg_df[self.kg_df['entity'] == text_h]['image'].unique().tolist()

        if random.random() < 0.8:
            selected_intra_class_images = random.choices(all_intra_class_images, k=self.num_intra_class_images)
        else: selected_intra_class_images = [self.umls_image_source_list[idx]] * self.num_intra_class_images

        #read all images and store in 3, 12, 96, 96 tensor
        intra_class_images = []
        for img in selected_intra_class_images:
            img = Image.open(img).convert("RGB")
            img = self.transform[1](img)
            intra_class_images.append(img)

        intra_class_images = torch.stack(intra_class_images)
        intra_class_images = intra_class_images.view(3, self.num_intra_class_images, intra_class_images.shape[-1], intra_class_images.shape[-1])
                
        if random.random() < 0.33:
            text = text_h
        elif random.random() < 0.66:
            text = text_r + ' ' + text_t
        else: text = def_h

        sample = {}
        sample['text'] = text
        sample['image'] = image
        sample['class'] = idx_class
        sample['intra_class_images'] = intra_class_images
        return sample
    
class UMLS_Dataset_PathVQA_test(Dataset):
    def __init__(self, args, knowledge_graph_path, transform):
        #read the json file
        with open(knowledge_graph_path, 'r') as f:
            self.kg = json.load(f)

        #for all items, get entities into an array
        self.source_image_list = []
        self.source_entity_list = []
        self.source_entity_def_list = []
        self.target_entity_list = []
        self.edge_list = []

        #self.preprocess = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32").image_processor
        #_, self.preprocess, _ = open_clip.create_model_and_transforms('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')

        self.num_intra_class_images = args.num_intra_class_images
        self.transform = transform

        for item in self.kg:
            if(item['def_head'] == None):
                continue

            self.source_image_list.append(os.path.join(args.image_source_folder, item['image']))
            self.source_entity_list.append(item['head_entity'].lower())
            self.target_entity_list.append(item['tail_entity'].lower())
            self.edge_list.append(item['relation'].lower())
            self.source_entity_def_list.append(item['def_head'].lower())

        self.kg_df = pd.DataFrame(columns=['image', 'entity', 'def', 'relation', 'related entity'])
        self.kg_df['image'] = self.source_image_list
        self.kg_df['entity'] = self.source_entity_list
        self.kg_df['def'] = self.source_entity_def_list
        self.kg_df['relation'] = self.edge_list
        self.kg_df['related entity'] = self.target_entity_list

        #self.umls_kg_info = pd.read_csv(umls_kg_file)
        self.umls_image_source_list = self.kg_df['image']
        self.umls_kg_source_list = self.kg_df['entity']
        self.umls_kg_target_list = self.kg_df['related entity']
        self.umls_kg_edge_list = self.kg_df['relation']
        self.umls_def_source_list = self.kg_df['def']

        self.classes = self.kg_df['entity'].unique()
        self.num_classes = len(self.classes)

        self.umls_data_len = self.kg_df.shape[0]

        #self.classes = self.kg_df['entity'].unique()

        self.cls_num_list = [len(self.kg_df[self.kg_df['entity'] == cls]) for cls in self.classes]

        print('UMLS data length: ', self.umls_data_len)
    
    def __len__(self):
        return int(self.umls_data_len)
    
    def __getitem__(self, idx):
        text_h = self.umls_kg_source_list[idx]
        def_h = self.umls_def_source_list[idx]
        text_t = self.umls_kg_target_list[idx]
        text_r = self.umls_kg_edge_list[idx]

        image = self.transform[0](Image.open(self.umls_image_source_list[idx]).convert("RGB"))
        intra_class_image_representation = self.transform[1](Image.open(self.umls_image_source_list[idx]).convert("RGB"))
        intra_class_image_representation = torch.stack([intra_class_image_representation for _ in range(self.num_intra_class_images)]).view(3, self.num_intra_class_images, intra_class_image_representation.shape[-1], intra_class_image_representation.shape[-1])
        relation = text_r + ' ' + text_t
        head =  text_h
        head_def = def_h

        #repeat intra_class_image_representation for num_intra_class_images times
        sample = {}
        sample['relation'] = relation
        sample['head'] = head
        sample['head_def'] = head_def
        sample['image'] = image
        sample['intra_class_images'] = intra_class_image_representation
        return sample
    
class UMLS_Dataset_VQARAD(Dataset):
    def __init__(self, args, knowledge_graph_path, transform):
        #read the json file
        with open(knowledge_graph_path, 'r') as f:
            self.kg = json.load(f)

        #for all items, get entities into an array
        self.source_image_list = []
        self.source_entity_list = []
        self.source_entity_def_list = []
        self.target_entity_list = []
        self.edge_list = []

        self.num_intra_class_images = args.num_intra_class_images
        self.transform = transform

        #self.preprocess = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32").image_processor
        #_, self.preprocess, _ = open_clip.create_model_and_transforms('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')

        for item in self.kg:
            if(item['def_head'] == None):
                continue

            self.source_image_list.append(os.path.join(args.image_source_folder, item['image']))
            self.source_entity_list.append(item['head_entity'].lower())
            self.target_entity_list.append(item['tail_entity'].lower())
            self.edge_list.append(item['relation'].lower())
            self.source_entity_def_list.append(item['def_head'].lower())

        self.kg_df = pd.DataFrame(columns=['image', 'entity', 'def', 'relation', 'related entity'])
        self.kg_df['image'] = self.source_image_list
        self.kg_df['entity'] = self.source_entity_list
        self.kg_df['def'] = self.source_entity_def_list
        self.kg_df['relation'] = self.edge_list
        self.kg_df['related entity'] = self.target_entity_list

        #self.umls_kg_info = pd.read_csv(umls_kg_file)
        self.umls_image_source_list = self.kg_df['image']
        self.umls_kg_source_list = self.kg_df['entity']
        self.umls_kg_target_list = self.kg_df['related entity']
        self.umls_kg_edge_list = self.kg_df['relation']
        self.umls_def_source_list = self.kg_df['def']

        self.classes = self.kg_df['entity'].unique()
        self.num_classes = len(self.classes)

        self.umls_data_len = self.kg_df.shape[0]

        #self.classes = self.kg_df['entity'].unique()

        self.cls_num_list = [len(self.kg_df[self.kg_df['entity'] == cls]) for cls in self.classes]

        print('UMLS data length: ', self.umls_data_len)
        print("Head class num: ", len(self.cls_num_list))
    
    def __len__(self):
        return int(self.umls_data_len)
    
    def __getitem__(self, idx):
        image = self.transform[0](Image.open(self.umls_image_source_list[idx])) #single image for classification.
        text_h = self.umls_kg_source_list[idx]
        def_h = self.umls_def_source_list[idx]
        text_t = self.umls_kg_target_list[idx]
        text_r = self.umls_kg_edge_list[idx]
        idx_class = np.where(self.classes == text_h)[0][0]

        all_intra_class_images = self.kg_df[self.kg_df['entity'] == text_h]['image'].unique().tolist()

        if random.random() < 0.8:
            selected_intra_class_images = random.choices(all_intra_class_images, k=self.num_intra_class_images)
        else: selected_intra_class_images = [self.umls_image_source_list[idx]] * self.num_intra_class_images

        #read all images and store in 3, 12, 96, 96 tensor
        intra_class_images = []
        for img in selected_intra_class_images:
            img = Image.open(img)
            img = self.transform[1](img)
            intra_class_images.append(img)

        intra_class_images = torch.stack(intra_class_images)
        intra_class_images = intra_class_images.view(3, self.num_intra_class_images, intra_class_images.shape[-1], intra_class_images.shape[-1])
                
        if random.random() < 0.33:
            text = text_h
        elif random.random() < 0.66:
            text = text_r + ' ' + text_t
        else: text = def_h

        sample = {}
        sample['text'] = text
        sample['image'] = image
        sample['class'] = idx_class
        sample['intra_class_images'] = intra_class_images
        return sample
    
class UMLS_Dataset_VQARAD_test(Dataset):
    def __init__(self, args, knowledge_graph_path, transform):
        #read the json file
        with open(knowledge_graph_path, 'r') as f:
            self.kg = json.load(f)

        #for all items, get entities into an array
        self.source_image_list = []
        self.source_entity_list = []
        self.source_entity_def_list = []
        self.target_entity_list = []
        self.edge_list = []

        self.num_intra_class_images = args.num_intra_class_images
        self.transform = transform

        #self.preprocess = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32").image_processor
        #_, self.preprocess, _ = open_clip.create_model_and_transforms('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')

        for item in self.kg:
            if(item['def_head'] == None):
                continue

            self.source_image_list.append(os.path.join(args.image_source_folder, item['image']))
            self.source_entity_list.append(item['head_entity'].lower())
            self.target_entity_list.append(item['tail_entity'].lower())
            self.edge_list.append(item['relation'].lower())
            self.source_entity_def_list.append(item['def_head'].lower())

        self.kg_df = pd.DataFrame(columns=['image', 'entity', 'def', 'relation', 'related entity'])
        self.kg_df['image'] = self.source_image_list
        self.kg_df['entity'] = self.source_entity_list
        self.kg_df['def'] = self.source_entity_def_list
        self.kg_df['relation'] = self.edge_list
        self.kg_df['related entity'] = self.target_entity_list

        #self.umls_kg_info = pd.read_csv(umls_kg_file)
        self.umls_image_source_list = self.kg_df['image']
        self.umls_kg_source_list = self.kg_df['entity']
        self.umls_kg_target_list = self.kg_df['related entity']
        self.umls_kg_edge_list = self.kg_df['relation']
        self.umls_def_source_list = self.kg_df['def']

        self.classes = self.kg_df['entity'].unique()
        self.num_classes = len(self.classes)

        self.umls_data_len = self.kg_df.shape[0]

        #self.classes = self.kg_df['entity'].unique()

        self.cls_num_list = [len(self.kg_df[self.kg_df['entity'] == cls]) for cls in self.classes]

        print('Test UMLS data length: ', self.umls_data_len)
        print('Test Head class num: ', len(self.cls_num_list))
    
    def __len__(self):
        return int(self.umls_data_len)
    
    def __getitem__(self, idx):
        text_h = self.umls_kg_source_list[idx]
        def_h = self.umls_def_source_list[idx]
        text_t = self.umls_kg_target_list[idx]
        text_r = self.umls_kg_edge_list[idx]

        image = self.transform[0](Image.open(self.umls_image_source_list[idx]))
        intra_class_image_representation = self.transform[1](Image.open(self.umls_image_source_list[idx]))
        intra_class_image_representation = torch.stack([intra_class_image_representation for _ in range(self.num_intra_class_images)]).view(3, self.num_intra_class_images, intra_class_image_representation.shape[-1], intra_class_image_representation.shape[-1])
        relation = text_r + ' ' + text_t
        head =  text_h
        head_def = def_h

        #repeat intra_class_image_representation for num_intra_class_images times
        sample = {}
        sample['relation'] = relation
        sample['head'] = head
        sample['head_def'] = head_def
        sample['image'] = image
        sample['intra_class_images'] = intra_class_image_representation
        return sample
    
""" class UMLS_Dataset_test(Dataset):
    def __init__(self, args):
        #read the json file
        with open(args.knowledge_graph, 'r') as f:
            self.kg = json.load(f)

        #for all items, get entities into an array
        self.source_image_list = []
        self.source_entity_list = []
        self.source_entity_def_list = []
        self.source_entity_cuis_list = []
        self.target_entity_list = []
        self.edge_list = []

        #self.preprocess = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32").image_processor
        _, self.preprocess, _ = open_clip.create_model_and_transforms('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')

        for item in self.kg:
            if item['Relations'] == None:
                continue

            for target in item['Relations']:
                self.source_image_list.append(os.path.join(args.image_source_folder, item['image']))
                self.source_entity_list.append(item['entity'])
                self.target_entity_list.append(target['related entity'])
                self.edge_list.append(target['relation'])

                self.source_entity_cuis_list.append(item['CUI'])
                self.source_entity_def_list.append(item['def'])

        self.kg_df = pd.DataFrame(columns=['image', 'entity', 'def', 'CUI', 'relation', 'related entity'])
        self.kg_df['image'] = self.source_image_list
        self.kg_df['entity'] = self.source_entity_list
        self.kg_df['def'] = self.source_entity_def_list
        self.kg_df['CUI'] = self.source_entity_cuis_list
        self.kg_df['relation'] = self.edge_list
        self.kg_df['related entity'] = self.target_entity_list

        #self.umls_kg_info = pd.read_csv(umls_kg_file)
        self.umls_image_source_list = self.kg_df['image']
        self.umls_kg_source_list = self.kg_df['entity']
        self.umls_kg_target_list = self.kg_df['related entity']
        self.umls_kg_edge_list = self.kg_df['relation']
        self.umls_cui_source_list = self.kg_df['CUI']
        self.umls_def_source_list = self.kg_df['def']

        self.umls_cui_target_list = self.kg_df['related entity']

        self.umls_data_len = self.kg_df.shape[0]

        print('UMLS data length: ', self.umls_data_len)
    
    def __len__(self):
        return int(self.umls_data_len)
    
    def __getitem__(self, idx):
        if("pathvqa" in self.umls_image_source_list[idx]): 
            path = "/mnt/storage1/ziya/VQA/Datasets/pathvqa/images_CLIP"
            self.umls_image_source_list[idx] = os.path.join(path, self.umls_image_source_list[idx].replace("pathvqa/", ""))
        image = self.preprocess(Image.open(self.umls_image_source_list[idx]))
        text_h = self.umls_kg_source_list[idx]
        #cui_h = self.umls_cui_source_list[idx]
        def_h = self.umls_def_source_list[idx]
        text_t = self.umls_kg_target_list[idx]
        #cui_t = self.umls_cui_target_list[idx]
        text_r = self.umls_kg_edge_list[idx]

        relation = text_r + ' [SEP] ' + text_t
        head =  text_h
        head_def = def_h

        sample = {}
        sample['relation'] = relation
        sample['head'] = head
        sample['head_def'] = head_def
        sample['image'] = image

        return sample """

        
class SharedEpoch:
    def __init__(self, epoch: int = 0):
        self.shared_epoch = Value('i', epoch)

    def set_value(self, epoch):
        self.shared_epoch.value = epoch

    def get_value(self):
        return self.shared_epoch.value

@dataclass
class DataInfo:
    dataloader: DataLoader
    sampler: DistributedSampler = None
    shared_epoch: SharedEpoch = None

    def set_epoch(self, epoch):
        if self.shared_epoch is not None:
            self.shared_epoch.set_value(epoch)
        if self.sampler is not None and isinstance(self.sampler, DistributedSampler):
            self.sampler.set_epoch(epoch)
