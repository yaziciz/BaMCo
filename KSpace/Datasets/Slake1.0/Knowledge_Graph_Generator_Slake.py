import spacy
from scispacy.linking import EntityLinker
import json
import torch
import networkx as nx
import matplotlib.pyplot as plt
import torch.nn.functional as F

from tqdm import tqdm
from collections import defaultdict
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

# Define split paths, replace with your actual paths
# The dataset can be downloaded from here: https://huggingface.co/datasets/BoKelvin/SLAKE
# The images should be downloaded seperately from the files section of the given repository above.
split_paths = {
    "train": "BaMCo/Datasets/Slake1.0/train.json",
    "test": "BaMCo/Datasets/Slake1.0/test.json",
    "validation": "BaMCo/Datasets/Slake1.0/validate.json"
}

for split_name, split_path in split_paths.items():
    # Read split data
    with open(split_path) as f:
        split_data = json.load(f)

    data = []
    for i in tqdm(range(len(split_data)), desc=f"Processing {split_name}"):
        entry = split_data[i]
        # Filter out unwanted content types and answers
        if entry["answer"].lower() == 'no': continue
        if entry["content_type"].lower() in ['color', 'size', 'shape', 'quantity', 'position']: continue
        if entry["answer"].lower() == 'yes':
            keywords = nlp(entry["question"]).ents
        else:
            keywords = nlp(entry["answer"]).ents

        for keyword in keywords:
            keyword = keyword.text
            entity_nlp = querry_UMLS(keyword)
            if entity_nlp is None or entity_nlp['Definition'] is None:
                continue
            data.append({
                'image': entry['img_name'],
                'entity': entity_nlp['Name'].lower(),
                'def': entity_nlp['Definition'],
                'CUI': entity_nlp['UI'],
                'Relations': entity_nlp['Relations']
            })

    # Save the knowledge graph data for this split
    with open(f'KG_Slake_{split_name}.json', 'w') as f:
        json.dump(data, f, indent=4)