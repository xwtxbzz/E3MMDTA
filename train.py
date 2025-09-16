import torch
from torch import nn
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from model import CoreNet
from torch_geometric.loader import DataLoader
from dataset import *
from utils import *
import time 
dataset = 'kiba'
data = pd.read_csv("/workspace/dataset/private/xyj/data/{}/affinity.csv".format(dataset))
data = data.to_numpy()[:]
train_data, test_data = train_test_split(data, test_size=0.16667, random_state=42)
print(train_data.shape,test_data.shape)
train = DTAData(train_data,dataset_name=dataset)
train_data = DataLoader(train,batch_size = 128,shuffle=True)
test = DTAData(test_data,dataset_name=dataset)
test_data = DataLoader(test,batch_size = 128)
model = CoreNet().cuda()
loss_fn = nn.MSELoss()
sl1 = nn.SmoothL1Loss()
l1  = nn.L1Loss()
huber = nn.HuberLoss()
# 0.005 0.1
# 0.0005 0.01
# 0.0001 0.0001
optimizer = torch.optim.SGD(model.parameters(), lr=0.0001, weight_decay=0.00005)
if os.path.exists('/workspace/algorithm/test/model/CoreNet_B_{}.pth'.format(dataset)):
    model.load_state_dict(torch.load('/workspace/algorithm/test/model/CoreNet_KAF_{}.pth'.format(dataset)))

torch.cuda.empty_cache()
avg_loss = 0
for epoch in range(500):
    print('Epoch:', epoch)
    avg_loss = 0
    torch.cuda.empty_cache()
    i = 0      
    model.train()  
    for s_smiles, s_sequence,value,e3fp,ergfp,pubfp,maccsfp,data_mol,data_pro,smi_embed,seq_embed in train_data:
        optimizer.zero_grad()
        ti = time.time()
        output = model(
                       s_smiles.cuda(),
                       s_sequence.cuda(),
                       e3fp.cuda(),
                       ergfp.cuda(),
                       pubfp.cuda(),
                       maccsfp.cuda(),
                       data_mol.cuda(),
                       data_pro.cuda(),
                       smi_embed.cuda(),
                       seq_embed.cuda(),
                    )
        loss = loss_fn(output, torch.FloatTensor(value).cuda())
        sl = l1(output, torch.FloatTensor(value).cuda())
        a_loss  =  loss
        avg_loss += loss.item()
        print('loss:', loss.item(), 'time:', time.time() - ti, 'i:', i)
        i += 1
        a_loss.backward()
        optimizer.step()
        torch.cuda.empty_cache()
        
    print('Average Loss:', avg_loss / len(train_data))
    torch.save(model.state_dict(), '/workspace/algorithm/test/model/CoreNet_KAF_{}.pth'.format(dataset))
    y = []
    p = []
    model.eval()
    for s_smiles, s_sequence, value,e3fp,ergfp,pubfp,maccsfp,data_mol,data_pro,smi_embed,seq_embed in test_data:
        torch.cuda.empty_cache()
        ti = time.time()
        with torch.no_grad():
            output = model(
                        s_smiles.cuda(),
                        s_sequence.cuda(),
                        e3fp.cuda(),
                        ergfp.cuda(),
                        pubfp.cuda(),
                        maccsfp.cuda(),
                        data_mol.cuda(),
                        data_pro.cuda(),
                        smi_embed.cuda(),
                        seq_embed.cuda()
                      )
        y.extend(value.flatten())
        p.extend(output.flatten().cpu().detach().numpy())
    calculate_metrics(np.array(y), np.array(p), dataset=dataset,type='test')