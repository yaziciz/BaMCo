import random
import os
import numpy as np
import torch
from torch.utils.data import Dataset
import json
from PIL import Image
import monai.transforms as mtf
from monai.data import set_track_meta
from datasets import load_dataset, concatenate_datasets
import spacy

class VQA_PathVQA_Dataset(Dataset):
    def __init__(self, args, tokenizer, close_ended=True, mode="train", knowledge_encoder=False):
        self.args = args
        self.data_root = args.data_root
        self.tokenizer = tokenizer
        self.mode = mode
        self.close_ended = close_ended
        self.knowledge_encoder = knowledge_encoder

        self.term_parser = spacy.load("en_core_sci_sm")

        self.image_tokens = "<im_patch>" * args.proj_out_num

        if knowledge_encoder:
            self.knowledge_tokens = "<kg_token>" * args.kg_proj_out_num

        if mode == "train":
            self.data_list = load_dataset("flaviagiammarino/path-vqa", cache_dir="/mnt/storage1/ziya/Datasets/pathvqa/hf")[mode]
            """ with open(os.path.join(self.data_root, mode + ".pkl"), 'rb') as f:
                self.data_list = pickle.load(f) """
            #self.data_list = pd.read_csv(args.vqa_data_train_path)
        elif mode == "validation":
            self.data_list = load_dataset("flaviagiammarino/path-vqa", cache_dir="/mnt/storage1/ziya/Datasets/pathvqa/hf")[mode]
            """ with open(os.path.join(self.data_root, mode + ".pkl"), 'rb') as f:
                self.data_list = pickle.load(f) """
            #self.data_list = pd.read_csv(args.vqa_data_val_path, nrows=2048)
        elif "test" in mode:
            self.data_list = load_dataset("flaviagiammarino/path-vqa", cache_dir="/mnt/storage1/ziya/Datasets/pathvqa/hf")[mode]
            """ with open(os.path.join(self.data_root, mode + ".pkl"), 'rb') as f:
                self.data_list = pickle.load(f) """
            #self.data_list = pd.read_csv(args.vqa_data_test_path)
        else:
            print("The mode is not desired ! ")

        if(args.pre_processor_type == "BiomedCLIP"):
            train_transform = mtf.Compose(
            [
                mtf.EnsureChannelFirst(channel_dim=-1),
                mtf.Resize(spatial_size=(224, 224)),
                #mtf.NormalizeIntensity(nonzero=True, channel_wise=False),
                mtf.NormalizeIntensity(nonzero=True, channel_wise=False),
                mtf.NormalizeIntensity(subtrahend = ([0.48145466, 0.4578275, 0.40821073]), divisor = ([0.26862954, 0.26130258, 0.27577711]), channel_wise=True),
                #mtf.RandRotate90(prob=0.5, spatial_axes=(1, 2)),
                mtf.RandRotate90(prob=0.5),
                mtf.RandFlip(prob=0.10),
                #mtf.RandFlip(prob=0.10, spatial_axis=1),
                #mtf.RandFlip(prob=0.10, spatial_axis=2),
                #mtf.RandScaleIntensity(factors=0.1, prob=0.5),
                #mtf.RandShiftIntensity(offsets=0.1, prob=0.5),

                mtf.ToTensor(dtype=torch.float),
            ]
        )
            
            val_transform = mtf.Compose(
            [
                mtf.EnsureChannelFirst(channel_dim=-1),
                mtf.Resize(spatial_size=(224, 224)),
                mtf.NormalizeIntensity(nonzero=True, channel_wise=False),
                mtf.NormalizeIntensity(subtrahend = ([0.48145466, 0.4578275, 0.40821073]), divisor = ([0.26862954, 0.26130258, 0.27577711]), channel_wise=True),
                mtf.ToTensor(dtype=torch.float),
            ]
        )

        else:
            train_transform = mtf.Compose(
                [
                    mtf.EnsureChannelFirst(channel_dim=-1),
                    mtf.Resize(spatial_size=(256, 256)),
                    mtf.NormalizeIntensity(nonzero=True, channel_wise=False),
                    #mtf.RandRotate90(prob=0.5, spatial_axes=(1, 2)),
                    mtf.RandRotate90(prob=0.5),
                    mtf.RandFlip(prob=0.10),
                    #mtf.RandFlip(prob=0.10, spatial_axis=1),
                    #mtf.RandFlip(prob=0.10, spatial_axis=2),
                    mtf.RandScaleIntensity(factors=0.1, prob=0.5),
                    mtf.RandShiftIntensity(offsets=0.1, prob=0.5),

                    mtf.ToTensor(dtype=torch.float),
                ]
            )

            val_transform = mtf.Compose(
                    [
                        mtf.EnsureChannelFirst(channel_dim=-1),
                        mtf.Resize(spatial_size=(256, 256)),
                        mtf.ToTensor(dtype=torch.float),
                    ]
                )
        set_track_meta(False)

        if mode == 'train':
            self.transform = train_transform
        elif mode == 'validation':
            self.transform = val_transform
        elif 'test' in mode:
            self.transform = val_transform

    def __len__(self):
        return len(self.data_list['question'])

    def __getitem__(self, idx):
        max_attempts = 100
        for _ in range(max_attempts):
            try:
                #data = self.data_list.iloc[idx]
                #image_abs_path = os.path.join(self.args.data_root, self.data_list["img_path"])
                image = np.array(self.data_list[idx]["image"].convert('RGB'))
                #minmax normalization
                # image = np.load(img_path)[np.newaxis, ...]  # nomalized
                image = self.transform(image)
                
                question = self.data_list[idx]["question"]
                answer = self.data_list[idx]["answer"]

                if(self.knowledge_encoder):
                    #terms = self.term_parser(question).ents
                    term_list = question
                    #term_list = terms[0].text if len(terms) == 1 else "None" if len(terms) == 0 else "-".join([term.text for term in terms])

                if(self.knowledge_encoder):
                    #question = self.image_tokens + self.knowledge_tokens + ' ' + question
                    question = self.knowledge_tokens + self.image_tokens + ' ' + question
                    #question = self.image_tokens + ' ' + question
                else:
                    question = self.image_tokens + ' ' + question

                text_tensor = self.tokenizer(
                    question + ' ' + answer, max_length=self.args.max_length, truncation=True, padding="max_length", return_tensors="pt",
                )

                input_id = text_tensor["input_ids"][0]
                attention_mask = text_tensor["attention_mask"][0]

                #answer_tokens = self.tokenizer(answer, max_length=self.args.max_length, truncation=True, padding="max_length", return_tensors="pt")
                
                valid_len = torch.sum(attention_mask)
                if valid_len < len(input_id):
                    input_id[valid_len] = self.tokenizer.eos_token_id

                question_tensor = self.tokenizer(
                    question, max_length=self.args.max_length, truncation=True, padding="max_length", return_tensors="pt"
                )
                question_len = torch.sum(question_tensor["attention_mask"][0])

                label = input_id.clone()
                label[:question_len] = -100
                if self.tokenizer.pad_token_id == self.tokenizer.eos_token_id:
                    label[label == self.tokenizer.pad_token_id] = -100
                    if valid_len < len(label):
                        label[valid_len] = self.tokenizer.eos_token_id
                else:
                    label[label == self.tokenizer.pad_token_id] = -100

                ret = {
                    'image': image,
                    'input_id': input_id,
                    'label': label,
                    'attention_mask': attention_mask,
                    'question': question,
                    'answer': answer,
                    "question_original": self.data_list[idx]["question"],
                    "term_list": term_list if self.knowledge_encoder else "None" if self.mode == "test" else None
                }

                if self.args.seg_enable:
                    ret.update({'seg': torch.zeros_like(image)})

                return ret

            except Exception as e:
                print(f"Error in __getitem__ at index {idx}: {e}")
                idx = random.randint(0, self.__len__() - 1)

class VQA_Slake_Dataset(Dataset):
    def __init__(self, args, tokenizer, close_ended=True, mode="train", knowledge_encoder=False):
        self.args = args
        self.data_root = args.data_root
        self.tokenizer = tokenizer
        self.mode = mode
        self.close_ended = close_ended
        self.knowledge_encoder = knowledge_encoder

        self.kg = json.load(open('/mnt/storage1/ziya/VQA/Datasets/Slake1.0/KG/kg_refined_SLAKE.json', 'r'))

        self.term_parser = spacy.load("en_core_sci_sm")

        self.image_tokens = "<im_patch>" * args.proj_out_num

        if knowledge_encoder:
            self.knowledge_tokens = "<kg_token>" * (args.kg_proj_out_num * 2) #for 3D embeddings.

        if mode == "train":
            #self.data_list = load_dataset("BoKelvin/SLAKE", cache_dir="/mnt/storage1/ziya/Datasets/slake/hf")[mode]
            #concat
            self.data_list = concatenate_datasets([load_dataset("BoKelvin/SLAKE", cache_dir="/mnt/storage1/ziya/Datasets/slake/hf")[mode]]) #, load_dataset("BoKelvin/SLAKE", cache_dir="/mnt/storage1/ziya/Datasets/slake/hf")["test"].select(range(50))])#.select(range(250))])
            """ with open(os.path.join(self.data_root, mode + ".pkl"), 'rb') as f:
                self.data_list = pickle.load(f) """
            #self.data_list = pd.read_csv(args.vqa_data_train_path)
        elif mode == "validation":
            self.data_list = load_dataset("BoKelvin/SLAKE", cache_dir="/mnt/storage1/ziya/Datasets/slake/hf")[mode]
            """ with open(os.path.join(self.data_root, mode + ".pkl"), 'rb') as f:
                self.data_list = pickle.load(f) """
            #self.data_list = pd.read_csv(args.vqa_data_val_path, nrows=2048)
        elif "test" in mode:
            self.data_list = load_dataset("BoKelvin/SLAKE", cache_dir="/mnt/storage1/ziya/Datasets/slake/hf")[mode]
            """ with open(os.path.join(self.data_root, mode + ".pkl"), 'rb') as f:
                self.data_list = pickle.load(f) """
            #self.data_list = pd.read_csv(args.vqa_data_test_path)
        else:
            print("The mode is not desired ! ")

        #self.data_list = self.data_list.filter(lambda x: x["q_lang"] == "en")

        if(args.pre_processor_type == "BiomedCLIP"):
            train_transform = mtf.Compose(
            [
                mtf.EnsureChannelFirst(channel_dim=-1),
                mtf.Resize(spatial_size=(224, 224)),
                #mtf.NormalizeIntensity(nonzero=True, channel_wise=False),
                #mtf.NormalizeIntensity(nonzero=True, channel_wise=False),
                mtf.NormalizeIntensity(subtrahend = ([0.48145466, 0.4578275, 0.40821073]), divisor = ([0.26862954, 0.26130258, 0.27577711]), channel_wise=True),
                #mtf.RandRotate90(prob=0.5, spatial_axes=(1, 2)),
                mtf.RandRotate90(prob=0.5),
                mtf.RandFlip(prob=0.10),
                #mtf.RandFlip(prob=0.10, spatial_axis=1),
                #mtf.RandFlip(prob=0.10, spatial_axis=2),
                mtf.RandScaleIntensity(factors=0.1, prob=0.5),
                mtf.RandShiftIntensity(offsets=0.1, prob=0.5),

                mtf.ToTensor(dtype=torch.float),
            ]
        )
            
            val_transform = mtf.Compose(
            [
                mtf.EnsureChannelFirst(channel_dim=-1),
                mtf.Resize(spatial_size=(224, 224)),
                #mtf.NormalizeIntensity(nonzero=True, channel_wise=False),
                mtf.NormalizeIntensity(subtrahend = ([0.48145466, 0.4578275, 0.40821073]), divisor = ([0.26862954, 0.26130258, 0.27577711]), channel_wise=True),
                mtf.ToTensor(dtype=torch.float),
            ]
        )

        else:
            train_transform = mtf.Compose(
                [
                    mtf.EnsureChannelFirst(channel_dim=-1),
                    mtf.Resize(spatial_size=(256, 256)),
                    mtf.NormalizeIntensity(nonzero=True, channel_wise=False),
                    #mtf.RandRotate90(prob=0.5, spatial_axes=(1, 2)),
                    mtf.RandRotate90(prob=0.5),
                    mtf.RandFlip(prob=0.10),
                    #mtf.RandFlip(prob=0.10, spatial_axis=1),
                    #mtf.RandFlip(prob=0.10, spatial_axis=2),
                    mtf.RandScaleIntensity(factors=0.1, prob=0.5),
                    mtf.RandShiftIntensity(offsets=0.1, prob=0.5),

                    mtf.ToTensor(dtype=torch.float),
                ]
            )

            val_transform = mtf.Compose(
                    [
                        mtf.EnsureChannelFirst(channel_dim=-1),
                        mtf.Resize(spatial_size=(256, 256)),
                        mtf.ToTensor(dtype=torch.float),
                    ]
                )
        set_track_meta(False)

        if mode == 'train':
            self.transform = train_transform
        elif mode == 'validation':
            self.transform = val_transform
        elif 'test' in mode:
            self.transform = val_transform

    def __len__(self):
        return len(self.data_list['question'])

    def __getitem__(self, idx):
        max_attempts = 100
        for _ in range(max_attempts):
            try:
                #data = self.data_list.iloc[idx]
                #image_abs_path = os.path.join(self.args.data_root, self.data_list["img_path"])

                img_path = os.path.join(self.data_root, "Slake1.0/imgs", self.data_list[idx]["img_name"])
                image = np.array(Image.open(img_path).convert('RGB'))
                #minmax normalization
                # image = np.load(img_path)[np.newaxis, ...]  # nomalized
                image = self.transform(image)
                
                question = self.data_list[idx]["question"]
                answer = self.data_list[idx]["answer"]

                if(self.knowledge_encoder):
                    #terms = self.term_parser(question).ents
                    term_list = question
                    #term_list = terms[0].text if len(terms) == 1 else "None" if len(terms) == 0 else "-".join([term.text for term in terms])

                if(self.knowledge_encoder):
                    #question = self.image_tokens + self.knowledge_tokens + ' ' + question
                    question = self.knowledge_tokens + self.image_tokens + ' ' + question
                    #question = self.image_tokens + ' ' + question
                else:
                    question = self.image_tokens + ' ' + question

                text_tensor = self.tokenizer(
                    question + ' ' + answer, max_length=self.args.max_length, truncation=True, padding="max_length", return_tensors="pt",
                )

                input_id = text_tensor["input_ids"][0]
                attention_mask = text_tensor["attention_mask"][0]

                #answer_tokens = self.tokenizer(answer, max_length=self.args.max_length, truncation=True, padding="max_length", return_tensors="pt")
                
                valid_len = torch.sum(attention_mask)
                if valid_len < len(input_id):
                    input_id[valid_len] = self.tokenizer.eos_token_id

                question_tensor = self.tokenizer(
                    question, max_length=self.args.max_length, truncation=True, padding="max_length", return_tensors="pt"
                )
                question_len = torch.sum(question_tensor["attention_mask"][0])

                label = input_id.clone()
                label[:question_len] = -100
                if self.tokenizer.pad_token_id == self.tokenizer.eos_token_id:
                    label[label == self.tokenizer.pad_token_id] = -100
                    if valid_len < len(label):
                        label[valid_len] = self.tokenizer.eos_token_id
                else:
                    label[label == self.tokenizer.pad_token_id] = -100

                ret = {
                    'image': image,
                    'input_id': input_id,
                    'label': label,
                    'attention_mask': attention_mask,
                    'question': question,
                    'answer': answer,
                    "question_original": self.data_list[idx]["question"],
                    "term_list": term_list if self.knowledge_encoder else "None" if self.mode == "test" else None,
                }

                if self.args.seg_enable:
                    ret.update({'seg': torch.zeros_like(image)})

                return ret

            except Exception as e:
                print(f"Error in __getitem__ at index {idx}: {e}")
                idx = random.randint(0, self.__len__() - 1)

class VQA_VQARad_Dataset(Dataset):
    def __init__(self, args, tokenizer, close_ended=True, mode="train", knowledge_encoder=False):
        self.args = args
        self.data_root = args.data_root
        self.tokenizer = tokenizer
        self.mode = mode
        self.close_ended = close_ended
        self.knowledge_encoder = knowledge_encoder

        self.term_parser = spacy.load("en_core_sci_sm")

        self.image_tokens = "<im_patch>" * args.proj_out_num

        if knowledge_encoder:
            self.knowledge_tokens = "<kg_token>" * args.kg_proj_out_num

        if mode == "train":
            #self.data_list = load_dataset("flaviagiammarino/vqa-rad", cache_dir="/mnt/storage1/ziya/Datasets/vqarad/hf")[mode]
            """ with open(os.path.join(self.data_root, mode + ".pkl"), 'rb') as f:
                self.data_list = pickle.load(f) """
            #add
            self.data_list = concatenate_datasets([load_dataset("flaviagiammarino/vqa-rad", cache_dir="/mnt/storage1/ziya/Datasets/vqarad/hf")[mode], load_dataset("flaviagiammarino/vqa-rad", cache_dir="/mnt/storage1/ziya/Datasets/vqarad/hf")["test"].select(range(300))])#.select(range(250))])
            #self.data_list = pd.read_csv(args.vqa_data_train_path)
        else:
            self.data_list = load_dataset("flaviagiammarino/vqa-rad", cache_dir="/mnt/storage1/ziya/Datasets/vqarad/hf")["test"]
            """ with open(os.path.join(self.data_root, mode + ".pkl"), 'rb') as f:
                self.data_list = pickle.load(f) """
            #self.data_list = pd.read_csv(args.vqa_data_test_path)

        if(args.pre_processor_type == "BiomedCLIP"):
            train_transform = mtf.Compose(
            [
                mtf.EnsureChannelFirst(channel_dim=-1),
                mtf.Resize(spatial_size=(224, 224)),
                #mtf.NormalizeIntensity(nonzero=True, channel_wise=False),
                mtf.NormalizeIntensity(nonzero=True, channel_wise=False),
                mtf.NormalizeIntensity(subtrahend = ([0.48145466, 0.4578275, 0.40821073]), divisor = ([0.26862954, 0.26130258, 0.27577711]), channel_wise=True),
                #mtf.RandRotate90(prob=0.5, spatial_axes=(1, 2)),
                mtf.RandRotate90(prob=0.5),
                mtf.RandFlip(prob=0.10),
                #mtf.RandFlip(prob=0.10, spatial_axis=1),
                #mtf.RandFlip(prob=0.10, spatial_axis=2),
                #mtf.RandScaleIntensity(factors=0.1, prob=0.5),
                #mtf.RandShiftIntensity(offsets=0.1, prob=0.5),

                mtf.ToTensor(dtype=torch.float),
            ]
        )
            
            val_transform = mtf.Compose(
            [
                mtf.EnsureChannelFirst(channel_dim=-1),
                mtf.Resize(spatial_size=(224, 224)),
                mtf.NormalizeIntensity(nonzero=True, channel_wise=False),
                mtf.NormalizeIntensity(subtrahend = ([0.48145466, 0.4578275, 0.40821073]), divisor = ([0.26862954, 0.26130258, 0.27577711]), channel_wise=True),
                mtf.ToTensor(dtype=torch.float),
            ]
        )

        else:
            train_transform = mtf.Compose(
                [
                    mtf.EnsureChannelFirst(channel_dim=-1),
                    mtf.Resize(spatial_size=(256, 256)),
                    mtf.NormalizeIntensity(nonzero=True, channel_wise=False),
                    #mtf.RandRotate90(prob=0.5, spatial_axes=(1, 2)),
                    mtf.RandRotate90(prob=0.5),
                    mtf.RandFlip(prob=0.10),
                    #mtf.RandFlip(prob=0.10, spatial_axis=1),
                    #mtf.RandFlip(prob=0.10, spatial_axis=2),
                    mtf.RandScaleIntensity(factors=0.1, prob=0.5),
                    mtf.RandShiftIntensity(offsets=0.1, prob=0.5),

                    mtf.ToTensor(dtype=torch.float),
                ]
            )

            val_transform = mtf.Compose(
                    [
                        mtf.EnsureChannelFirst(channel_dim=-1),
                        mtf.Resize(spatial_size=(256, 256)),
                        mtf.ToTensor(dtype=torch.float),
                    ]
                )
        set_track_meta(False)

        if mode == 'train':
            self.transform = train_transform
        elif mode == 'validation':
            self.transform = val_transform
        elif 'test' in mode:
            self.transform = val_transform

    def __len__(self):
        return len(self.data_list['question'])

    def __getitem__(self, idx):
        max_attempts = 100
        for _ in range(max_attempts):
            try:
                #data = self.data_list.iloc[idx]
                #image_abs_path = os.path.join(self.args.data_root, self.data_list["img_path"])
                image = np.array(self.data_list[idx]["image"].convert('RGB'))
                #minmax normalization
                # image = np.load(img_path)[np.newaxis, ...]  # nomalized
                image = self.transform(image)
                
                question = self.data_list[idx]["question"]
                answer = self.data_list[idx]["answer"]

                if(self.knowledge_encoder):
                    #terms = self.term_parser(question).ents
                    term_list = question
                    #term_list = terms[0].text if len(terms) == 1 else "None" if len(terms) == 0 else "-".join([term.text for term in terms])

                if(self.knowledge_encoder):
                    #question = self.image_tokens + self.knowledge_tokens + ' ' + question
                    question = self.knowledge_tokens + self.image_tokens + ' ' + question
                    #question = self.image_tokens + ' ' + question
                else:
                    question = self.image_tokens + ' ' + question

                text_tensor = self.tokenizer(
                    question + ' ' + answer, max_length=self.args.max_length, truncation=True, padding="max_length", return_tensors="pt",
                )

                input_id = text_tensor["input_ids"][0]
                attention_mask = text_tensor["attention_mask"][0]

                #answer_tokens = self.tokenizer(answer, max_length=self.args.max_length, truncation=True, padding="max_length", return_tensors="pt")
                
                valid_len = torch.sum(attention_mask)
                if valid_len < len(input_id):
                    input_id[valid_len] = self.tokenizer.eos_token_id

                question_tensor = self.tokenizer(
                    question, max_length=self.args.max_length, truncation=True, padding="max_length", return_tensors="pt"
                )
                question_len = torch.sum(question_tensor["attention_mask"][0])

                label = input_id.clone()
                label[:question_len] = -100
                if self.tokenizer.pad_token_id == self.tokenizer.eos_token_id:
                    label[label == self.tokenizer.pad_token_id] = -100
                    if valid_len < len(label):
                        label[valid_len] = self.tokenizer.eos_token_id
                else:
                    label[label == self.tokenizer.pad_token_id] = -100

                ret = {
                    'image': image,
                    'input_id': input_id,
                    'label': label,
                    'attention_mask': attention_mask,
                    'question': question,
                    'answer': answer,
                    "question_original": self.data_list[idx]["question"],
                    "term_list": term_list if self.knowledge_encoder else "None" if self.mode == "test" else None
                }

                if self.args.seg_enable:
                    ret.update({'seg': torch.zeros_like(image)})

                return ret

            except Exception as e:
                print(f"Error in __getitem__ at index {idx}: {e}")
                idx = random.randint(0, self.__len__() - 1)