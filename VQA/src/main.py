import os
import logging
from typing import Optional, List, Dict
import numpy as np
import torch
import transformers
from transformers import AutoTokenizer, LlamaForCausalLM, GPT2Tokenizer
from dataclasses import dataclass, field
from dataset.multi_dataset import VQA_PathVQA_Dataset, CapDataset,  TextDatasets, UniDatasets, VQA_Slake_Dataset, VQA_VQARad_Dataset
from model.language_model import BaMCoLlamaForCausalLM, LamedPhi3ForCausalLM, BaMCoGPT2ForCausalLM
from train.BaMCo_VQA_trainer import BaMCoVQATrainer
from huggingface_hub import login
from tqdm import tqdm
import evaluate
import json
import wandb

from transformers import BitsAndBytesConfig, Trainer
from huggingface_hub import hf_hub_download, HfApi
api = HfApi()

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


local_rank = None
tokenizer = None
login(token="<your HuggingFace token>") # Replace with your Hugging Face token
wandb.login(key="<your WandB key>") # Replace with your WandB key

def rank0_print(*args):
    if local_rank == 0:
        print(*args)

@dataclass
class ModelArguments:
    version: Optional[str] = field(default="v0")
    model_name_or_path: Optional[str] = field(default="meta-llama/Llama-3.2-1B", metadata={"help": "Path to the LLM or MLLM."}) #"meta-llama/Llama-3.2-1B", "openai-community/gpt2-xl"
    model_type: Optional[str] = field(default="llama3", metadata={"help": "llama3, phi3, gpt2"})

    freeze_backbone: bool = field(default=True)
    pretrain_mllm: Optional[str] = field(default=None)

    tune_mm_mlp_adapter: bool = field(default=False, metadata={"help": "Used in pretrain: tune mm_projector and embed_tokens"})
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None, metadata={"help": "Path to pretrained mm_projector and embed_tokens."})

    eval_model_path: Optional[str] = field(default="BaMCo/VQA/src/outputs", metadata={"help": "Path to the model for evaluation."})

    # image
    image_channel: int = field(default=3)
    image_size: tuple = field(default=(224, 224))
    patch_size: tuple = field(default=(16, 16))

    #eval
    max_new_tokens: int = field(default=256)
    do_sample: bool = field(default=False)
    top_p: float = field(default=None)
    temperature: float = field(default=1.0)

    # vision
    vision_tower: Optional[str] = field(default="BiomedCLIP") # None, "vit", "BiomedCLIP"
    vision_select_layer: Optional[int] = field(default=-1)
    vision_select_feature: Optional[str] = field(default="patch")
    pretrain_vision_model: str = field(default=None, metadata={"help": "Path to pretrained model for ViT."})
    freeze_vision_tower: bool = field(default=True)

    #knowledge encoder
    knowledge_encoder: bool = field(default=True)
    knowledge_encoder_checkpoint: Optional[str] = field(default="BaMCo/KSpace/src/checkpoints/Slake_KnowledgeSpace.pt")
    freeze_knowledge_encoder: bool = field(default=True)
    #kg_dim: int = field(default=256)
    #kg_pool_size: int = field(default=32)

    # projector
    mm_projector_type: Optional[str] = field(default='spp', metadata={"help": "spp"}) #image projection layer.
    proj_layer_type: str = field(default="mlp", metadata={"help": "Type of layer in projector. options: [linear, mlp]."})
    proj_layer_num: int = field(default=2, metadata={"help": "Number of layers in projector."})
    proj_pooling_type: str = field(default="spatial", metadata={"help": "Type of pooling in projector. options: [spatial, sequence]."})
    proj_pooling_size: int = field(default=2, metadata={"help": "Size of pooling in projector."})

    # segvol
    segmentation_module: str = field(default=None, metadata={"help": "segvol"})
    pretrain_seg_module: str = field(default=None, metadata={"help": "Pretrained segvol model."})


@dataclass
class DataArguments:
    data_root: str = field(default="BaMCo/KSpace/Datasets", metadata={"help": "Root directory for all data."})

@dataclass
class TrainingArguments(transformers.TrainingArguments):

    # If the model is required to be evaluated only, set eval_only to True.
    # This will skip the training phase and directly evaluate the model.
    # If you want to train the model, set eval_only to False.
    # Following the training, the model will be evaluated on the test set.
    eval_only: bool = field(default=False)

    # lora
    lora_enable: bool = True
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0 #0.005
    #lora_weight_path: str = ""
    #lora_bias: str = "none"

    cache_dir: Optional[str] = "BaMCo/cache"
    remove_unused_columns: bool = field(default=False)
    model_max_length: int = field(
        default=256, #512
        metadata={
            "help":
            "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    seed: int = 25 
    #ddp_backend: str = "nccl"
    #ddp_timeout: int = 128000
    #ddp_find_unused_parameters: bool = False
    optim: str = field(default="adamw_torch")

    # This is set up to facilitate debugging, pls config these in bash file in training.
    bf16: bool = True
    output_dir: str = "BaMCo/VQA/src/outputs/Llama3.2_1B_Slake"
    run_name: str = "Llama3.2_1B_Slake"
    num_train_epochs: float = 5
    per_device_train_batch_size: int = 100
    per_device_eval_batch_size: int = 100
    gradient_accumulation_steps: int = 1
    eval_strategy: str = "epoch"
    eval_accumulation_steps: int = 1
    eval_steps: float = 0.1
    save_strategy: str = "epoch"
    save_steps: int = 0.1
    save_total_limit: int = 2
    learning_rate: float = 0.0001 #1e-4
    weight_decay: float = 0 #0.005
    warmup_ratio: float = 0.1 
    lr_scheduler_type: str = "cosine"
    logging_steps: float = 100
    
    gradient_checkpointing: bool = False # train fast
    dataloader_pin_memory: bool = True # fast
    dataloader_num_workers: int = 16 
    
    auto_find_batch_size: bool = True
    load_best_model_at_end: bool = True
    #report_to: str = "tensorboard"
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False

def preprocess_logits_for_metrics(logits, labels):
    pred_ids = torch.argmax(logits, dim=-1)
    return pred_ids

def find_all_linear_names(model):
    if("gpt2" in model.name_or_path):
        cls = transformers.pytorch_utils.Conv1D
        model = model.base_model
    elif("Llama-3.2" in model.name_or_path):
        cls = torch.nn.Linear
    else: print("Unknown model type")

    lora_module_names = set()
    # Process of elimination: LoRA only targets on LLM backbone
    ignore_keywords = ['vision_tower', 'mm_projector', 'embed_tokens', 'lm_head', 'seg_projector', 'seg_module']
    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in ignore_keywords):
            continue
        if isinstance(module, cls):
            lora_module_names.add(name)
    return list(lora_module_names)

@dataclass
class DataCollator:
    def __init__(self, seg_enable):
        self.seg_enable = seg_enable
    def __call__(self, batch: list) -> dict:
        if self.seg_enable:
            images, input_ids, labels, attention_mask, segs, term_list = tuple(
                [b[key] for b in batch] for key in ('image', 'input_id', 'label', 'attention_mask', 'seg', 'term_list'))

            images = torch.cat([_.unsqueeze(0) for _ in images], dim=0)
            input_ids = torch.cat([_.unsqueeze(0) for _ in input_ids], dim=0)
            labels = torch.cat([_.unsqueeze(0) for _ in labels], dim=0)
            attention_mask = torch.cat([_.unsqueeze(0) for _ in attention_mask], dim=0)

            for i, seg in enumerate(segs):
                if seg.sum() == 0:
                    segs[i] = torch.zeros((1, 1, 32, 256, 256))
                else:
                    segs[i] = seg.unsqueeze(0)
            segs = torch.cat(segs, dim=0)

            return_dict = dict(
                images=images,
                input_ids=input_ids,
                labels=labels,
                attention_mask=attention_mask,
                segs=segs,
                term_list=term_list,
            )
        else:
            images, input_ids, labels, attention_mask, term_list = tuple(
                [b[key] for b in batch] for key in ('image', 'input_id', 'label', 'attention_mask', 'term_list'))

            images = torch.cat([_.unsqueeze(0) for _ in images], dim=0)
            input_ids = torch.cat([_.unsqueeze(0) for _ in input_ids], dim=0)
            labels = torch.cat([_.unsqueeze(0) for _ in labels], dim=0)
            attention_mask = torch.cat([_.unsqueeze(0) for _ in attention_mask], dim=0)

            return_dict = dict(
                images=images,
                input_ids=input_ids,
                labels=labels,
                attention_mask=attention_mask,
                term_list=term_list
            )

        return return_dict

def main():
    global local_rank
    global tokenizer

    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    local_rank = training_args.local_rank
    
    seed = training_args.seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    rank0_print("="*20 + " Tokenizer preparation " + "="*20)
    # Load tokenizer from the given path with specified configurations

    if 'gpt2' in model_args.model_type:
        tokenizer = GPT2Tokenizer.from_pretrained(model_args.model_name_or_path) 
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right",
            use_fast=False,
        )

    # Define and add special tokens
    if model_args.knowledge_encoder:
        special_token = {"additional_special_tokens": ["<im_patch>", "<kg_token>", "<bx_start>", "<bx_end>"]}
        model_args.num_new_tokens = 5
    else: 
        special_token = {"additional_special_tokens": ["<im_patch>", "<bx_start>", "<bx_end>"]}
        model_args.num_new_tokens = 4

    tokenizer.add_special_tokens(
        special_token
    )
    tokenizer.add_tokens("[SEG]")

    if tokenizer.unk_token is not None and tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.unk_token

    if("gpt2" in model_args.model_type):
        tokenizer.eos_token_id = 50256
        
    if 'llama3' in model_args.model_type:
        tokenizer.eos_token_id = 128001
        tokenizer.pad_token = tokenizer.eos_token

    # Convert special tokens to token IDs and set related arguments
    model_args.img_token_id = tokenizer.convert_tokens_to_ids("<im_patch>")
    model_args.seg_token_id = tokenizer.convert_tokens_to_ids("[SEG]")

    if model_args.knowledge_encoder: model_args.kg_token_id = tokenizer.convert_tokens_to_ids("<kg_token>")

    model_args.vocab_size = len(tokenizer)
    rank0_print("seg_token_id: ", model_args.seg_token_id)
    rank0_print("vocab_size: ", model_args.vocab_size)

    rank0_print("="*20 + " Model preparation " + "="*20)
    
    if model_args.vision_tower is not None:
        if 'llama' in model_args.model_type:
            model = BaMCoLlamaForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                bos_token_id=tokenizer.bos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        elif 'phi3' in model_args.model_type:
            model = LamedPhi3ForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir
                )
        elif 'gpt2' in model_args.model_type:
            model = BaMCoGPT2ForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                bos_token_id=tokenizer.bos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        else:
            raise ValueError(f"Unknown Model Type {model_args.model_type}")
    else:
        model = LlamaForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir
        )

    if model_args.freeze_backbone:
        model.requires_grad_(False)

    model.config.tune_mm_mlp_adapter = training_args.tune_mm_mlp_adapter = model_args.tune_mm_mlp_adapter
    if model_args.tune_mm_mlp_adapter:
        model.requires_grad_(False)
        for p in model.get_model().mm_projector.parameters():
            p.requires_grad = True

    if training_args.lora_enable:
        from peft import LoraConfig, get_peft_model
        lora_config = LoraConfig(
            r=training_args.lora_r,
            lora_alpha=training_args.lora_alpha,
            target_modules=find_all_linear_names(model),
            lora_dropout=training_args.lora_dropout,
            #bias=training_args.lora_bias,
            task_type="CAUSAL_LM",
        )

        rank0_print("Adding LoRA adapters only on LLM.")
        model = get_peft_model(model, lora_config)

    
    model.config.seg_token_id = model_args.seg_token_id
    model.config.use_cache = False

    model.enable_input_require_grads()
    if training_args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    # initialize vision and seg modules on LLM
    if model_args.vision_tower is not None:
        model.get_model().initialize_vision_modules(model_args=model_args)
    if model_args.segmentation_module is not None:
        model.get_model().initialize_seg_modules(model_args=model_args)
    if model_args.knowledge_encoder:
        model.get_model().initialize_knowledge_module(model_args=model_args)

    if model_args.pretrain_mllm:
        ckpt = torch.load(model_args.pretrain_mllm, map_location="cpu")
        model.load_state_dict(ckpt, strict=True)
        rank0_print("load pretrained MLLM weights.")

    model.initialize_vision_tokenizer(model_args, tokenizer) #output lm_head is disabled.

    model.print_trainable_parameters()

    # ckpt = torch.load("PATH/model_with_lora.bin", map_location="cpu")
    # model.load_state_dict(ckpt, strict=True)

    rank0_print("="*20 + " Dataset preparation " + "="*20)
    data_args.max_length = training_args.model_max_length
    data_args.proj_out_num = model.get_model().mm_projector.proj_out_num

    if model_args.knowledge_encoder:
        data_args.kg_proj_out_num = model.get_model().kg_projector.proj_out_num

    if(model_args.vision_tower == "BiomedCLIP"):
        data_args.pre_processor_type = "BiomedCLIP"
    else: data_args.pre_processor_type = "Default"

    rank0_print("vision tokens output from projector: ", data_args.proj_out_num)

    if model_args.knowledge_encoder:
        rank0_print("knowledge tokens output from projector: ", data_args.kg_proj_out_num)
    data_args.seg_enable = hasattr(model.get_model(), "seg_module")
    
    train_dataset = VQA_Slake_Dataset(data_args, tokenizer, mode='train', knowledge_encoder=model_args.knowledge_encoder)
    eval_dataset = VQA_Slake_Dataset(data_args, tokenizer, mode='validation', knowledge_encoder=model_args.knowledge_encoder) #CapDataset(data_args, tokenizer, mode='validation')
    test_dataset = VQA_Slake_Dataset(data_args, tokenizer, mode='test', knowledge_encoder=model_args.knowledge_encoder) #CapDataset(data_args, tokenizer, mode='validation')
    data_collator = DataCollator(data_args.seg_enable)

    if(not training_args.eval_only):
        rank0_print("="*20 + " Training " + "="*20)

        trainer = BaMCoVQATrainer(
                                model=model,
                                args=training_args,
                                data_collator=data_collator,
                                train_dataset=train_dataset,
                                eval_dataset=eval_dataset,
                                processing_class=tokenizer,
                                callbacks=[transformers.EarlyStoppingCallback(4)],
                      )

        trainer.train()

        rank0_print("="*20 + " Saving the Best Model " + "="*20)
        torch.save(model.state_dict(), os.path.join(training_args.output_dir, 'pytorch_model_best.bin'))

    else: 
        rank0_print("="*20 + " Loading and Pushing to the Hub " + "="*20)
        state_dict = torch.load(os.path.join(training_args.output_dir, 'pytorch_model_best.bin'), map_location="cuda", weights_only=True)
        model.load_state_dict(state_dict, strict=True)

        # Push the model to the Hugging Face Hub before the evaluation phase
        """ try:
            api.create_repo("<your_repo_name>/" + training_args.run_name, private=True)
            api.upload_file( # Upload in the background (non-blocking action)
                repo_id="<your_repo_name>/" + training_args.run_name,
                path_or_fileobj=os.path.join(training_args.output_dir, 'pytorch_model_best.bin'),
                repo_type="model",
                path_in_repo="pytorch_model_best.bin",
            )
        except:
            print("Repo already exists. Skipping the upload step...") """
        #model = LamedLlamaForCausalLM.from_pretrained("<your_repo_name>/" + training_args.run_name)

    rank0_print("="*20 + " Evaluate on Test Set " + "="*20)
    evaluater(model, test_dataset, training_args, model_args, data_args)

def evaluater(model, dataset, training_args, model_args, data_args):
    # Evaluation metrics are initialized from the Hugging Face Evaluate library
    bleu = evaluate.load("bleu")
    bertscore = evaluate.load("bertscore")
    meteor = evaluate.load("meteor")
    rouge = evaluate.load("rouge")

    model.to("cuda")
    model.eval()

    from torch.utils.data import DataLoader
    test_dataloader = DataLoader(
            dataset,
            batch_size=1,
            num_workers=2,
            pin_memory=True,
            shuffle=False,
            drop_last=False,
    )
    
    if not os.path.exists(training_args.output_dir):
        os.makedirs(training_args.output_dir)
    
    eval_json_path = os.path.join(training_args.output_dir, "eval.json")

    result_bleu = 0
    result_rouge = 0
    result_meteor = 0
    result_bert = 0
    result_accuracy_oe = 0
    result_accuracy_ce = 0

    c_ce = 0 # count of closed-ended questions
    c_oe = 0 # count of open-ended questions

    outputs = []
    
    # Iterate through the test dataset
    for sample in tqdm(test_dataloader):
        question = sample["question"]
        answer = sample['answer']

        image = sample["image"].to(device="cuda")
        input_id = tokenizer(question, return_tensors="pt")['input_ids'].to(device="cuda")
        attention_mask = torch.ones((input_id.shape[0], input_id.shape[1])).to(device="cuda")
        
        #in term_list, convert all "None" to None
        for i, term in enumerate(sample["term_list"]):
            if term == "None":
                sample["term_list"][i] = None

        with torch.inference_mode():
            #The knowledge encoder is used in the generate function of the model.
            generation = model.generate(image, input_id, max_new_tokens=model_args.max_new_tokens,
                                        do_sample=model_args.do_sample, top_p=model_args.top_p,
                                        temperature=model_args.temperature, pad_token_id=tokenizer.eos_token_id,
                                        attention_mask=attention_mask, term_list=sample["term_list"])
            
        generated_texts = tokenizer.batch_decode(generation, skip_special_tokens=True)

        result = dict()
        decoded_preds, decoded_labels = postprocess_text(generated_texts, answer)

        if("yes" in decoded_labels[0] or "no" in decoded_labels[0]):
            c_ce += 1

            if(decoded_labels[0] == decoded_preds):
                result_accuracy_ce += 1
        
            outputs.append({"Question": sample["question_original"], "Answer": answer[0], "Prediction": decoded_preds, "Result ACC": decoded_labels[0][0] == decoded_preds[0]})
            
        else:
            try:
                c_oe += 1

                doesItContain = doesContain(decoded_preds, decoded_labels[0])

                if(doesItContain):
                    result_accuracy_oe += 1

                bleu_score = bleu.compute(predictions=decoded_preds, references=decoded_labels, max_order=1)
                result["bleu"] = bleu_score['bleu']

                rouge_score = rouge.compute(predictions=decoded_preds, references=decoded_labels, rouge_types=['rouge1'])
                result["rouge1"] = rouge_score['rouge1']

                meteor_score = meteor.compute(predictions=decoded_preds, references=decoded_labels)
                result["meteor"] = meteor_score['meteor']

                bert_score = bertscore.compute(predictions=decoded_preds, references=decoded_labels, lang="en")
                result["bert_f1"] = sum(bert_score['f1']) / len(bert_score['f1'])

                outputs.append({"Question": sample["question_original"], "Answer": answer[0], "Prediction": decoded_preds[0], "Result ACC": doesItContain, "Bleu": result["bleu"], "Rouge": result["rouge1"], "Meteor": result["meteor"], "Bert": result["bert_f1"]})

                result_bleu += bleu_score['bleu']
                result_rouge += rouge_score['rouge1']
                result_meteor += meteor_score['meteor']
                result_bert += result["bert_f1"]
            except:
                print("Error in calculating metrics")
                continue

    result_bleu /= c_oe
    result_rouge /= c_oe
    result_meteor /= c_oe
    result_bert /= c_oe
    result_accuracy_oe /= c_oe
    result_accuracy_ce /= c_ce

    outputs.append({"Bleu": result_bleu, "Rouge": result_rouge, "Meteor": result_meteor, "Bert": result_bert, "Accuracy OE": result_accuracy_oe, "Accuracy CE": result_accuracy_ce})

    print(outputs)

    #save json
    with open(eval_json_path, 'w') as f:
        json.dump(outputs, f, indent=4)

def postprocess_text(preds, labels):
    preds = [pred.strip().lower() for pred in preds]
    labels = [[label.strip().lower()] for label in labels]
    return preds, labels

def doesContain(sub_answer_texts, sub_correct_texts):
    """
    Check if any of the sub_answer_texts is contained in any of the sub_correct_texts.
    Args:
        sub_answer_texts (list): List of sub-answer texts.
        sub_correct_texts (list): List of sub-correct texts.
    Returns:
        bool: True if any sub-answer text is contained in any sub-correct text, False otherwise.
    """
    sub_answer_texts = sub_answer_texts[0].lower().split(',')
    for sub_answer_text in sub_answer_texts:
        for sub_correct_text in sub_correct_texts:
            try:
              if sub_answer_text[0] == ' ': 
                  sub_answer_text = sub_answer_text[1:]
              if sub_correct_text[0] == ' ': 
                  sub_correct_text = sub_correct_text[1:]
            except:
              pass
            if sub_answer_text in sub_correct_text or sub_correct_text in sub_answer_text:
                return True
    return False

if __name__ == "__main__":
    main()
