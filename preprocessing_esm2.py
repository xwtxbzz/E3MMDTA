import torch
import pandas as pd
import numpy as np
from utils import target_to_graph, smile_to_graph, CHARPROTSET, CHARISOSMISET, label_chars
dataset_name = 'davis'
seq = pd.read_csv("./data/{}/sequence.csv".format(dataset_name))
smi = pd.read_csv("./data/{}/smiles.csv".format(dataset_name))
model, alphabet = torch.hub.load("facebookresearch/esm:main", "esm2_t33_650M_UR50D")   
batch_converter = alphabet.get_batch_converter()
torch.cuda.empty_cache()
seqs = seq.iloc[0:,[0,1]].to_numpy()
model.eval()
for s in seqs:
    sequence = s[1][:min(len(s[1]),512)]
    name = s[0].split('|')[-1]
    data = [(name,sequence)]
    _, _, batch_tokens = batch_converter(data)
    model = model.cuda()
    batch_tokens = batch_tokens.cuda()
    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[33], return_contacts=True)
        target_size, target_features, target_edge_index = target_to_graph(s[1], results["contacts"].cpu().numpy()[0])
        np.savez('./data/davis/{}/{}.npz'.format('protein_graph', name), target_features=target_features, target_edge_index=target_edge_index)