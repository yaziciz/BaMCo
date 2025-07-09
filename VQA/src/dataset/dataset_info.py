import os, json

data_root = 'BaMCo/KSpace/Datasets'

def get_RAG_classes_dict(dataset):

    """
    Retrieves the number of unique head entities for the specified dataset.
    """

    if dataset == "Slake":
        kg = json.load(open('BaMCo/KSpace/Datasets/Slake1.0/KG_Slake_Train.json', 'r'))
        path_append = "Slake1.0/imgs"

    elif dataset == "PathVQA":
        kg = json.load(open('BaMCo/KSpace/Datasets/PathVQA/KG_PathVQA_Train.json', 'r'))
        path_append = "PathVQA/images"
    
    elif dataset == "VQARAD":
        kg = json.load(open('BaMCo/KSpace/Datasets/VQARAD/KG_VQARAD_Train.json', 'r'))
        path_append = "VQARAD/images"

    RAG_classes_dict = {}
    for entity in kg:
        if entity['head_entity'].lower() not in RAG_classes_dict:
            RAG_classes_dict[entity['head_entity'].lower()] = []
        #check str in [str]

        image_path = os.path.join(data_root, path_append, entity['image'])
        if  image_path not in RAG_classes_dict[entity['head_entity'].lower()]:
            RAG_classes_dict[entity['head_entity'].lower()].append(image_path)

    num_classes = len(list(RAG_classes_dict.keys()))
    RAG_classes_dict["num_classes"] = num_classes

    return RAG_classes_dict
