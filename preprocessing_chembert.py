import torch
import pandas as pd
import numpy as np
from utils import target_to_graph, smile_to_graph, CHARPROTSET, CHARISOSMISET, label_chars

import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import torch
from transformers import AutoModelForMaskedLM,AutoTokenizer
model = AutoModelForMaskedLM.from_pretrained("DeepChem/ChemBERTa-77M-MLM")
model.resize_token_embeddings(1280)    #
tokenizer = AutoTokenizer.from_pretrained("DeepChem/ChemBERTa-77M-MLM")
dataset = 'metz'
drug = pd.read_csv("./data/{}/smiles.csv".format(dataset)).to_numpy()
for s in drug:
    inputs = tokenizer(s[1], padding=True, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        print(outputs.logits.shape)
    smiles_size, smiles_features, smiles_edge_index = smile_to_graph(s[1])
    np.savez('./data/metz/{}/{}.npz'.format('drug_graph', s[0]), smiles_features=smiles_features, smiles_edge_index=smiles_edge_index,smiles_m = outputs.logits.cpu().numpy())
