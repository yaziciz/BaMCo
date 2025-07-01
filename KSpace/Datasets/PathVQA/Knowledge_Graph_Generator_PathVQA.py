import spacy
from scispacy.linking import EntityLinker
import json
import torch
import networkx as nx
import matplotlib.pyplot as plt
import torch.nn.functional as F

from tqdm import tqdm
from collections import defaultdict
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
        r = requests.get(full_url,params=query)
        r.raise_for_status()
        r.encoding = 'utf-8'
        outputs  = r.json()
        item = (([outputs['result']])[0])['results'][0]
    except Exception as except_error:
        return None
        
    #based on UI, get definition
    try:
        query = {'apiKey':apikey}
        r = requests.get(uri+'/rest/content/'+version+'/CUI/'+item['ui']+'/definitions',params=query)
        r.raise_for_status()
        r.encoding = 'utf-8'
        results = r.json()
        definition = None
        for result in results['result']:
            if result['rootSource'] == 'NCI' or result['rootSource'] == 'MSH':
                definition = result['value'].lower()
                break
    except Exception as except_error:
        definition = None

    #relations
    try:
        query = {'apiKey':apikey, 'sabs':'SNOMEDCT_US'}
        r = requests.get(uri+'/rest/content/'+version+'/CUI/'+item['ui']+'/relations',params=query)
        r.raise_for_status()
        r.encoding = 'utf-8'
        results = r.json()
        results['result'] = [result for result in results['result'] if result['additionalRelationLabel'] not in ["inverse_isa", "mapped_from", "mapped_to", "inverse_was_a", ]]
    except Exception as except_error:
        results = None
        relations = None

    if results is not None:
        relations = []
        for result in results['result']:
            relations.append({'relation': result['additionalRelationLabel'].lower().replace("_", " "), 'related entity': result['relatedIdName'].lower()})

    return {'UI': item['ui'], 'Name': item['name'], 'Definition': definition, 'Relations': relations}

# Load spaCy model
nlp = spacy.load("en_core_sci_sm")

# Load all splits at once
dataset = load_dataset("flaviagiammarino/path-vqa")
splits = {
    "train": dataset["train"],
    "test": dataset["test"],
    "validation": dataset["validation"]
}

# Directory to save images
image_save_dir = "BaMCo/Datasets/PathVQA/images"
os.makedirs(image_save_dir, exist_ok=True)

for split_name, DATASET in splits.items():
    data = []
    id = -1
    temp_image = None

    for i in tqdm(range(len(DATASET)), desc=f"Processing {split_name}"):
        if(DATASET[i]["answer"] == "no"): continue
        if DATASET[i]["answer"] == "yes":
            keywords = nlp(DATASET[i]["question"]).ents
        else:
            keywords = nlp(DATASET[i]["answer"]).ents

        if(temp_image != DATASET[i]["image"]):
            temp_image = DATASET[i]["image"]
            id += 1

            # Save PIL image to disk
            temp_image.save(f"{image_save_dir}{id}_{split_name}.jpg")

        for keyword in keywords:
            keyword = keyword.text
            entity_nlp = querry_UMLS(keyword)
            if(entity_nlp is None or entity_nlp['Definition'] == None): continue

            data.append({
                'image': f"{id}_{split_name}.jpg",
                'entity': entity_nlp['Name'].lower(),
                'def': entity_nlp['Definition'],
                'CUI': entity_nlp['UI'],
                'Relations': entity_nlp['Relations']
            })

    # Save the knowledge graph data for this split
    with open(f'KG_PathVQA_{split_name}.json', 'w') as f:
        json.dump(data, f, indent=4)