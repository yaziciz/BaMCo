import time
import spacy
import json
import torch
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from PIL import Image
import requests
import os

def querry_UMLS(string = "antifungal therapy"):
    apikey = "<your UMLS API key>"  # Replace with your UMLS API key
    version = "current"
    uri = "https://uts-ws.nlm.nih.gov"
    content_endpoint = "/rest/search/"+version
    full_url = uri+content_endpoint
    
    try:
        query = {'string':string,'apiKey':apikey, 'sabs':'SNOMEDCT_US'}
        r = requests.get(full_url,params=query, timeout=5)
        r.raise_for_status()
        r.encoding = 'utf-8'
        outputs  = r.json()
        item = (([outputs['result']])[0])['results'][0]
    except Exception:
        return None
        
    #based on UI, get definition
    try:
        query = {'apiKey':apikey}
        r = requests.get(uri+'/rest/content/'+version+'/CUI/'+item['ui']+'/definitions',params=query, timeout=5)
        r.raise_for_status()
        r.encoding = 'utf-8'
        results = r.json()
        definition = None
        for result in results['result']:
            if result['rootSource'] == 'NCI' or result['rootSource'] == 'MSH':
                definition = result['value'].lower()
                break
    except Exception:
        return None

    #relations
    try:
        query = {'apiKey':apikey, 'sabs':'SNOMEDCT_US'}
        r = requests.get(uri+'/rest/content/'+version+'/CUI/'+item['ui']+'/relations',params=query, timeout=5)
        r.raise_for_status()
        r.encoding = 'utf-8'
        results = r.json()
        results['result'] = [result for result in results['result'] if result['additionalRelationLabel'] not in ["inverse_isa", "mapped_from", "mapped_to", "inverse_was_a", ]]
    except Exception:
        return None

    if results is not None:
        relations = []
        for result in results['result']:
            relations.append({'relation': result['additionalRelationLabel'].lower().replace("_", " "), 'related entity': result['relatedIdName'].lower()})

    return {'UI': item['ui'], 'Name': item['name'], 'Definition': definition, 'Relations': relations}

# Load spaCy model
nlp = spacy.load("en_core_sci_sm")

# Define splits and output paths
splits = {
    "train": {
        "data": load_dataset("flaviagiammarino/vqa-rad")["train"],
        "img_dir": "BaMCo/Datasets/VQARAD/images/",
        "output_json": "BaMCo/Datasets/VQARAD/KG_VQARAD_Train.json"
    },
    "test": {
        "data": load_dataset("flaviagiammarino/vqa-rad")["test"],
        "img_dir": "BaMCo/Datasets/VQARAD/images/",
        "output_json": "BaMCo/Datasets/VQARAD/KG_VQARAD_Test.json"
    }
}

for split_name, split_info in splits.items():
    split_data = split_info["data"]
    img_dir = split_info["img_dir"]
    output_json = split_info["output_json"]

    os.makedirs(img_dir, exist_ok=True)
    data = []
    id = -1
    temp_image = None

    for i in tqdm(range(len(split_data)), desc=f"Processing {split_name}"):
        entry = split_data[i]
        if entry["answer"] == "no":
            continue
        if entry["answer"] == "yes":
            keywords = nlp(entry["question"]).ents
        else:
            keywords = nlp(entry["answer"]).ents

        for keyword in keywords:
            keyword = keyword.text
            entity_nlp = querry_UMLS(keyword)
            if (
                entity_nlp is None or
                entity_nlp['Definition'] is None or
                entity_nlp['Relations'] is None or
                entity_nlp['Name'] is None or
                entity_nlp['Name'] == "" or
                entity_nlp['Definition'] == "" or
                entity_nlp['Relations'] == []
            ):
                continue

            if temp_image != entry["image"]:
                temp_image = entry["image"]
                id += 1
                # Save PIL image to disk
                temp_image.save(os.path.join(img_dir, f"{id}_{split_name}.jpg"))

            # If entity has more than 5 relations, randomly select 5 (weighted)
            if len(entity_nlp['Relations']) > 5:
                relations = [rel['relation'] for rel in entity_nlp['Relations']]
                relations = np.array(relations)
                weights = np.array([relations.tolist().count(relation) for relation in relations])
                weights = 1 / weights
                weights = weights / np.sum(weights)
                chosen_indices = np.random.choice(len(entity_nlp['Relations']), 5, replace=False, p=weights)
                entity_nlp['Relations'] = [entity_nlp['Relations'][index] for index in chosen_indices]

            for relation in entity_nlp['Relations']:
                new_kg_item = {
                    'image': f"{id}_{split_name}.jpg",
                    'question': entry['question'].lower(),
                    'answer': entry['answer'].lower(),
                    'head_entity': entity_nlp['Name'].lower(),
                    'tail_entity': relation['related entity'],
                    'def_head': entity_nlp['Definition'],
                    'relation': relation['relation'],
                }
                data.append(new_kg_item)

    with open(output_json, 'w') as f:
        json.dump(data, f, indent=4)