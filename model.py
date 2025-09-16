import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.nn import GCNConv,SAGEConv,GATConv,SuperGATConv,GINConv,global_mean_pool as gep
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SuperGATConv, global_mean_pool as gep
class MultiModalFusion(nn.Module):
    def __init__(self, input_dims, hidden_dim=512, output_dim=256, num_heads=4, dropout=0.1):
        super(MultiModalFusion, self).__init__()
        self.input_dims = input_dims
        self.modality_names = list(input_dims.keys())
        self.modality_transform = nn.ModuleDict()
        for modality, dim in input_dims.items():
            self.modality_transform[modality] = nn.Sequential(
                nn.Linear(dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.LayerNorm(hidden_dim)
            )
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim, 
            num_heads=num_heads, 
            dropout=dropout,
        )
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim * len(input_dims), hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
        
        self.gates = nn.ModuleDict()
        for modality in self.modality_names:
            self.gates[modality] = nn.Sequential(
                nn.Linear(hidden_dim, 1),
                nn.Sigmoid()
            )
    
    def forward(self, modality_features):
        transformed_features = {}
        for modality in self.modality_names:
            transformed_features[modality] = self.modality_transform[modality](
                modality_features[modality]
            )
        stacked_features = torch.stack([
            transformed_features[modality] for modality in self.modality_names
        ], dim=1)
        attended_features, _ = self.cross_attention(
            stacked_features, stacked_features, stacked_features
        )
        gate_weights = torch.cat([
            self.gates[modality](transformed_features[modality]) 
            for modality in self.modality_names
        ], dim=1)
        
        gate_weights = F.softmax(gate_weights, dim=1)
        weighted_features = torch.stack([
            gate_weights[:, i].unsqueeze(1) * transformed_features[modality]
            for i, modality in enumerate(self.modality_names)
        ], dim=1)
        fused_features = torch.cat([
            weighted_features[:, i, :] for i in range(weighted_features.size(1))
        ], dim=1)
        output = self.output_proj(fused_features)
        return output

class CoreNet(nn.Module):
    def __init__(self, n_output=1, output_dim=256, dropout=0.1, 
                 num_features_mol=78, num_features_pro=33):
        super(CoreNet, self).__init__()
        self.mol_conv1 = GCNConv(num_features_mol, num_features_mol)
        self.mol_conv2 = GATConv(num_features_mol*2, num_features_mol * 2)
        self.mol_conv3 = SuperGATConv(num_features_mol * 4, num_features_mol * 4)
        self.mol_fc_g1 = nn.Linear(num_features_mol * 4, 256)
        self.mol_fc_g2 = nn.Linear(256, output_dim)

        self.pro_conv1 = GCNConv(num_features_pro, num_features_pro)
        self.pro_conv2 = GATConv(num_features_pro*2, num_features_pro * 2)
        self.pro_conv3 = SuperGATConv(num_features_pro * 4, num_features_pro * 4)
        self.pro_fc_g1 = nn.Linear(num_features_pro * 4, 256)
        self.pro_fc_g2 = nn.Linear(256, output_dim)

        self.sm = SimpleNet(embed=True, num_features=1024, output_dim=output_dim)
        self.se = SimpleNet(embed=True, num_features=1024, output_dim=output_dim)
        self.q = SimpleNet(num_features=1024, output_dim=output_dim)
        self.d = SimpleNet(num_features=1024, output_dim=output_dim)
        
        self.f1 = nn.Linear(1024, output_dim)  # e3fp
        self.f2 = nn.Linear(315, output_dim)   # ergfp
        self.f3 = nn.Linear(881, output_dim)   # pubfp
        self.f4 = nn.Linear(166, output_dim)   # maccsfp
        
        fusion_input_dims = {
            'mol_graph': output_dim,
            'pro_graph': output_dim,
            'smiles_seq': output_dim,
            'protein_seq': output_dim,
            'smi_m': output_dim,
            'seq_m': output_dim,
            'e3fp': output_dim,
            'ergfp': output_dim,
            'pubfp': output_dim,
            'maccsfp': output_dim
        }
        
        self.fusion_module = MultiModalFusion(
            input_dims=fusion_input_dims,
            hidden_dim=512,
            output_dim=output_dim * 2, 
            num_heads=4,
        )
        
        self.fc1 = nn.Linear(output_dim * 2, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.out = nn.Linear(512, n_output)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.bn1 = nn.BatchNorm1d(1024)
        self.bn2 = nn.BatchNorm1d(512)
        self.bn_out = nn.BatchNorm1d(n_output)
    
    def forward(self, smiles, sequence, e3fp, ergfp, pubfp, maccsfp, data_mol, data_pro, smi_m, seq_m):
        mol_x, mol_edge_index, mol_batch = data_mol.x, data_mol.edge_index, data_mol.batch
        x1 = self.mol_conv1(mol_x, mol_edge_index)
        x1 = self.relu(x1)
        
        x2 = self.mol_conv2(torch.cat([x1, mol_x], 1), mol_edge_index)
        x2 = self.relu(x2)
        
        x3 = self.mol_conv3(torch.cat([x1, mol_x, x2], 1), mol_edge_index)
        x3 = self.relu(x3)
        x3 = gep(x3, mol_batch)
        
        x = self.relu(self.mol_fc_g1(x3))
        x = self.dropout(x)
        mol_graph_feat = self.mol_fc_g2(x)
        mol_graph_feat = self.dropout(mol_graph_feat)

        target_x, target_edge_index, target_batch = data_pro.x, data_pro.edge_index, data_pro.batch
        xt1 = self.pro_conv1(target_x, target_edge_index)
        xt1 = self.relu(xt1)
        
        xt2 = self.pro_conv2(torch.cat([xt1, target_x], 1), target_edge_index)
        xt2 = self.relu(xt2)
        
        xt3 = self.pro_conv3(torch.cat([xt1, target_x, xt2], 1), target_edge_index)
        xt3 = self.relu(xt3)
        xt3 = gep(xt3, target_batch)
        
        xt = self.relu(self.pro_fc_g1(xt3))
        xt = self.dropout(xt)
        pro_graph_feat = self.pro_fc_g2(xt)
        pro_graph_feat = self.dropout(pro_graph_feat)

        smiles_seq_feat = self.sm(smiles)
        protein_seq_feat = self.se(sequence)
        smi_m_feat = self.q(smi_m)
        seq_m_feat = self.d(seq_m)
        
        e3fp_feat = self.f1(e3fp)
        ergfp_feat = self.f2(ergfp)
        pubfp_feat = self.f3(pubfp)
        maccsfp_feat = self.f4(maccsfp)
        
        modality_features = {
            'mol_graph': mol_graph_feat,
            'pro_graph': pro_graph_feat,
            'smiles_seq': smiles_seq_feat,
            'protein_seq': protein_seq_feat,
            'smi_m': smi_m_feat,
            'seq_m': seq_m_feat,
            'e3fp': e3fp_feat,
            'ergfp': ergfp_feat,
            'pubfp': pubfp_feat,
            'maccsfp': maccsfp_feat
        }
        
        fused_features = self.fusion_module(modality_features)
        
        xc = self.bn1(self.fc1(fused_features))
        xc = self.relu(xc)
        xc = self.dropout(xc)
        
        xc = self.bn2(self.fc2(xc))
        xc = self.relu(xc)
        xc = self.dropout(xc)
        
        out = self.out(xc)
        return self.bn_out(out)
