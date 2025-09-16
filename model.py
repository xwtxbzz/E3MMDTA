import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.nn import GCNConv,SAGEConv,GATConv,SuperGATConv,GINConv,global_mean_pool as gep
import math
class DynamicChunking(nn.Module):
    """动态分块模块：学习序列的分割边界"""
    def __init__(self, embed_dim, chunk_dim):
        super().__init__()
        self.conv = nn.Conv1d(embed_dim, chunk_dim, kernel_size=3, padding=1)
        self.gate = nn.Linear(chunk_dim, 1)  # 边界预测门控

    def forward(self, x):
        # x: (batch, seq_len, embed_dim)
        conv_out = self.conv(x.transpose(1, 2)).transpose(1, 2)  # (batch, seq_len, chunk_dim)
        gate_scores = torch.sigmoid(self.gate(conv_out))  # (batch, seq_len, 1)
        
        # 动态分块：门控>0.5的位置作为分块边界
        boundaries = (gate_scores.squeeze(-1) > 0.5).float()
        return conv_out, boundaries

class HierarchicalBlock(nn.Module):
    """层级处理块：对每个分块进行抽象表示"""
    def __init__(self, input_dim, hidden_dim,num_layers=2):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim if i==0 else hidden_dim, hidden_dim),
                nn.GELU()
            ) for i in range(num_layers)
        ])
        self.gate = nn.Linear(hidden_dim, input_dim)  # Highway门控:cite[6]

    def forward(self, x):
        residual = x
        for layer in self.layers:
            x = layer(x)
        gate = torch.sigmoid(self.gate(x))
        return gate * x + (1 - gate) * residual  # 门控残差连接

class HNet(nn.Module):
    """完整的H-Net模型"""
    def __init__(self, vocab_size=256, embed_dim=128, chunk_dim=32, num_levels=3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        # 多级处理模块
        self.levels = nn.ModuleList()
        for i in range(num_levels):
            self.levels.append(nn.Sequential(
                DynamicChunking(embed_dim if i==0 else chunk_dim, chunk_dim),
                HierarchicalBlock(chunk_dim, chunk_dim)
            ))
        
        self.output = nn.Linear(chunk_dim, embed_dim)

    def forward(self, x):
        # x: (batch, seq_len)
        x = self.embedding(x)  # (batch, seq_len, embed_dim)
        
        chunk_boundaries = []
        for level in self.levels:
            x, boundaries = level[0](x)  # 动态分块
            x = level[1](x)  # 层级抽象
            chunk_boundaries.append(boundaries)
        
        logits = self.output(x)  # (batch, seq_len, vocab_size)
        return logits
class SimpleNet(torch.nn.Module):
    def __init__(self, num_features=1024, output_dim=64, dropout=0.2
                 ,embed = False,drug = False):
        super(SimpleNet, self).__init__()
        print('SimpleNet Loaded')
        self.em = embed
        self.one = nn.Sequential(nn.Conv1d(num_features, output_dim, 1),nn.ReLU(),nn.Dropout(dropout))
        self.three = nn.Sequential(nn.Conv1d(num_features, output_dim, 3,padding=1),nn.ReLU(),nn.Dropout(dropout))
        self.five = nn.Sequential(nn.Conv1d(num_features, output_dim, 5,padding=2),nn.ReLU(),nn.Dropout(dropout))
        # self.seven = nn.Sequential(nn.Conv1d(num_features, output_dim, 7,padding=3),nn.LeakyReLU(),nn.Dropout(dropout))
        self.embed = nn.Embedding(num_features, output_dim)
        # self.hnet = HNet(num_features,output_dim)
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.BatchNorm1d(output_dim)
        self.out = nn.Linear(output_dim*3, output_dim)
        self.pool = nn.AvgPool1d(output_dim)
        self.l1 = nn.Linear(1280,output_dim)
        self.re = nn.ReLU()
        self.o_t = nn.Sequential(nn.Conv1d(output_dim, output_dim, 3,padding=1),nn.ReLU(),nn.Dropout(dropout))
        self.o_f = nn.Sequential(nn.Conv1d(output_dim, output_dim, 5,padding=2),nn.ReLU(),nn.Dropout(dropout))
        self.t_f = nn.Sequential(nn.Conv1d(output_dim, output_dim, 5,padding=2),nn.ReLU(),nn.Dropout(dropout))
    def forward(self,seq):
        if self.em:
            sq = self.embed(seq)
            # sq = self.hnet(seq)
        else:
            sq = self.re(self.l1(seq))
        o = self.o_f(self.o_t(self.one(sq)))
        t = self.t_f(self.three(sq))
        f = self.five(sq)
        x = torch.cat([o,t,f],dim=1)
        x = self.ln(x.permute(0,2,1))
        x = self.pool(self.out(x)).squeeze()
        return x
class RandomFourierFeatures(nn.Module):
    """
    Random Fourier Features (RFF)模块
    实现论文中的公式(2)
    """
    def __init__(self, input_dim, num_features, sigma=1.64):
        super().__init__()
        self.input_dim = input_dim
        self.num_features = num_features
        self.sigma = sigma
        
        # 初始化频率矩阵W (d x m)
        self.W = nn.Parameter(torch.randn(input_dim, num_features) * (sigma / math.sqrt(input_dim)))
        
        # 初始化相位偏移b (m,)
        self.b = nn.Parameter(torch.rand(num_features) * 2 * math.pi)
        
    def forward(self, x):
        # 计算⟨x, W⟩ + b
        x_proj = torch.matmul(x, self.W) + self.b  # (batch_size, num_features)
        
        # 计算cos和sin并拼接
        cos_part = torch.cos(x_proj)  # (batch_size, num_features)
        sin_part = torch.sin(x_proj)  # (batch_size, num_features)
        
        # 拼接并缩放
        output = torch.cat([cos_part, sin_part], dim=-1)  # (batch_size, 2*num_features)
        output = output * math.sqrt(2.0 / self.num_features)  # 缩放因子√(2/m)
        
        return output

class KAF_Layer(nn.Module):
    """
    KAF单层实现
    实现论文中的公式(1)
    """
    def __init__(self, input_dim, output_dim, rff_features=64, sigma=1.64):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.rff_features = rff_features
        
        # LayerNorm
        self.layernorm = nn.LayerNorm(input_dim)
        
        # GELU激活函数
        self.gelu = nn.GELU()
        
        # Random Fourier Features
        self.rff = RandomFourierFeatures(input_dim, rff_features, sigma)
        
        # RFF后的投影层V (2*rff_features -> input_dim)
        self.V = nn.Linear(2 * rff_features, input_dim)
        
        # 可学习的缩放参数a和b
        self.a = nn.Parameter(torch.ones(input_dim))  # 初始化为1
        self.b = nn.Parameter(torch.ones(input_dim) * 0.01)  # 初始化为0.01
        
        # 最终的线性变换层
        self.linear = nn.Linear(input_dim, output_dim)
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        # 初始化投影层V (Xavier初始化)
        nn.init.xavier_uniform_(self.V.weight)
        nn.init.zeros_(self.V.bias)
        
        # 初始化线性层
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)
    
    def forward(self, x):
        # LayerNorm
        x_norm = self.layernorm(x)  # (batch_size, input_dim)
        
        # GELU部分
        gelu_out = self.gelu(x_norm)  # (batch_size, input_dim)
        
        # RFF部分
        rff_out = self.rff(x_norm)  # (batch_size, 2*rff_features)
        rff_proj = self.V(rff_out)  # (batch_size, input_dim)
        
        # 混合激活: a ⊙ GELU(x) + b ⊙ φ̃(x)
        hybrid_activation = self.a * gelu_out + self.b * rff_proj  # (batch_size, input_dim)
        
        # 最终线性变换
        output = self.linear(hybrid_activation)  # (batch_size, output_dim)
        
        return output

class KAF_Network(nn.Module):
    """
    完整的KAF网络，由多个KAF层组成
    """
    def __init__(self, input_dim, hidden_dims, output_dim, rff_features=128, sigma=1.64):
        super().__init__()
        self.layers = nn.ModuleList()
        
        # 创建隐藏层
        dims = [input_dim] + hidden_dims + [output_dim]
        for i in range(len(dims) - 1):
            self.layers.append(
                KAF_Layer(dims[i], dims[i+1], rff_features, sigma)
            )
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
# class CoreNet(nn.Module):
#     def __init__(self, n_output=1, output_dim=256, dropout=0.1,num_features_mol=78,num_features_pro=33):
#         super(CoreNet, self).__init__()
        
        
        
#         self.mol_conv1 = GCNConv(num_features_mol, num_features_mol)
#         self.mol_conv2 = GATConv(num_features_mol*2, num_features_mol * 2)
#         self.mol_conv3 = SuperGATConv(num_features_mol * 4, num_features_mol * 4)
#         self.mol_fc_g1 = nn.Linear(num_features_mol * 4, 256)
#         self.mol_fc_g2 = nn.Linear(256, output_dim)

#         # self.pro_conv1 = GCNConv(embed_dim, embed_dim)
#         self.pro_conv1 = GCNConv(num_features_pro, num_features_pro)
#         self.pro_conv2 = GATConv(num_features_pro*2, num_features_pro * 2)
#         self.pro_conv3 = SuperGATConv(num_features_pro * 4, num_features_pro * 4)
#         # self.pro_conv4 = GCNConv(embed_dim * 4, embed_dim * 8)
#         self.pro_fc_g1 = nn.Linear(num_features_pro * 4, 256)
#         self.pro_fc_g2 = nn.Linear(256, output_dim)

#         self.relu = nn.ReLU()
#         self.dropout = nn.Dropout(dropout)
#         self.sm = SimpleNet(embed=True,num_features=1024,output_dim=output_dim)
#         self.se = SimpleNet(embed=True,num_features=1024,output_dim=output_dim)
#         self.q = SimpleNet(num_features=1024,output_dim=output_dim)
#         self.d = SimpleNet(num_features=1024,output_dim=output_dim)
#         self.fc1 =  nn.Linear(10*output_dim, 1024)
#         self.fc2 =  nn.Linear(1024, 512)
#         self.out =  nn.Linear(512,n_output)
#         self.relu = nn.ReLU()
#         self.dropout = nn.Dropout(dropout)
#         self.bn1 = nn.BatchNorm1d(1024)
#         self.bn2 = nn.BatchNorm1d(512)
#         self.bn = nn.BatchNorm1d(n_output)
#         self.f1 = nn.Linear(1024, output_dim)
#         self.f2 = nn.Linear(315, output_dim)
#         self.f3 = nn.Linear(881, output_dim)
#         self.f4 = nn.Linear(166, output_dim)
#     def forward(self,smiles,sequence,e3fp,ergfp,pubfp,maccsfp,data_mol,data_pro,smi_m,seq_m):
#         mol_x, mol_edge_index, mol_batch = data_mol.x, data_mol.edge_index, data_mol.batch
#         # get protein input
#         target_x, target_edge_index, target_batch = data_pro.x, data_pro.edge_index, data_pro.batch

#         x1 = self.mol_conv1(mol_x, mol_edge_index)
#         x1 = self.relu(x1)

#         # mol_edge_index, _ = dropout_adj(mol_edge_index, training=self.training)
#         x2 = self.mol_conv2(torch.cat([x1,mol_x],1), mol_edge_index)
#         x2 = self.relu(x2)

#         # mol_edge_index, _ = dropout_adj(mol_edge_index, training=self.training)
#         x3 = self.mol_conv3(torch.cat([x1,mol_x,x2],1), mol_edge_index)
#         x3 = self.relu(x3)
#         x3 = gep(x3, mol_batch)  # global pooling
#         # flatten
#         x = self.relu(self.mol_fc_g1(x3))
#         x = self.dropout(x)
#         x = self.mol_fc_g2(x)
#         x = self.dropout(x)

#         xt1 = self.pro_conv1(target_x, target_edge_index)
#         xt1 = self.relu(xt1)

#         # target_edge_index, _ = dropout_adj(target_edge_index, training=self.training)
#         xt2 = self.pro_conv2(torch.cat([xt1,target_x],1), target_edge_index)
#         xt2 = self.relu(xt2)

#         # target_edge_index, _ = dropout_adj(target_edge_index, training=self.training)
#         xt3 = self.pro_conv3(torch.cat([xt1,target_x,xt2],1), target_edge_index)
#         xt3 = self.relu(xt3)

#         # xt = self.pro_conv4(xt, target_edge_index)
#         # xt = self.relu(xt)
#         xt = gep(xt3, target_batch)  # global pooling

#         # flatten
#         xt = self.relu(self.pro_fc_g1(xt))
#         xt = self.dropout(xt)
#         xt = self.pro_fc_g2(xt)
#         xt = self.dropout(xt)

#         sex = self.se(sequence)
#         smx = self.sm(smiles)
#         sqx = self.q(seq_m)
#         sdx = self.d(smi_m)
#         xc = torch.cat((sex,smx,x,xt,sqx,sdx), 1)
        
#         f1 = self.f1(e3fp)
#         f2 = self.f2(ergfp)
#         f3 = self.f3(pubfp)
#         f4 = self.f4(maccsfp)
#         f = torch.cat([f1,f2,f3,f4],dim=1)
#         xc = torch.cat((xc,f), 1)
        
#         xc = self.bn1(self.fc1(xc))
#         xc = self.relu(xc)
#         xc = self.dropout(xc)
#         xc = self.bn2(self.fc2(xc))
#         xc = self.relu(xc)
#         xc = self.dropout(xc)
#         out = self.out(xc)
#         return self.bn(out)
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SuperGATConv, global_mean_pool as gep

class MultiModalFusion(nn.Module):
    def __init__(self, input_dims, hidden_dim=512, output_dim=256, num_heads=4, dropout=0.1):
        """
        多模态融合模块
        
        参数:
            input_dims: 各模态输入维度字典
            hidden_dim: 隐藏层维度
            output_dim: 输出维度
            num_heads: 多头注意力头数
            dropout: dropout率
        """
        super(MultiModalFusion, self).__init__()
        
        # 各模态的维度
        self.input_dims = input_dims
        self.modality_names = list(input_dims.keys())
        
        # 各模态的特征变换层
        self.modality_transform = nn.ModuleDict()
        for modality, dim in input_dims.items():
            self.modality_transform[modality] = nn.Sequential(
                nn.Linear(dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.LayerNorm(hidden_dim)
            )
        
        # 跨模态注意力融合
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim, 
            num_heads=num_heads, 
            dropout=dropout,
        )
        
        # 输出层
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim * len(input_dims), hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # 门控融合机制
        self.gates = nn.ModuleDict()
        for modality in self.modality_names:
            self.gates[modality] = nn.Sequential(
                nn.Linear(hidden_dim, 1),
                nn.Sigmoid()
            )
    
    def forward(self, modality_features):
        """
        前向传播
        
        参数:
            modality_features: 字典，包含各模态特征
            
        返回:
            融合后的特征向量
        """
        # 对各模态特征进行变换
        transformed_features = {}
        for modality in self.modality_names:
            transformed_features[modality] = self.modality_transform[modality](
                modality_features[modality]
            )
        
        # 准备注意力输入 (batch_size, num_modalities, hidden_dim)
        stacked_features = torch.stack([
            transformed_features[modality] for modality in self.modality_names
        ], dim=1)
        
        # 跨模态注意力
        attended_features, _ = self.cross_attention(
            stacked_features, stacked_features, stacked_features
        )
        
        # 门控加权融合
        gate_weights = torch.cat([
            self.gates[modality](transformed_features[modality]) 
            for modality in self.modality_names
        ], dim=1)
        
        gate_weights = F.softmax(gate_weights, dim=1)
        
        # 应用门控权重
        weighted_features = torch.stack([
            gate_weights[:, i].unsqueeze(1) * transformed_features[modality]
            for i, modality in enumerate(self.modality_names)
        ], dim=1)
        
        # 拼接所有特征
        fused_features = torch.cat([
            weighted_features[:, i, :] for i in range(weighted_features.size(1))
        ], dim=1)
        
        # 输出投影
        output = self.output_proj(fused_features)
        
        return output

class CoreNet(nn.Module):
    def __init__(self, n_output=1, output_dim=256, dropout=0.1, 
                 num_features_mol=78, num_features_pro=33):
        super(CoreNet, self).__init__()
        
        # 分子图分支
        self.mol_conv1 = GCNConv(num_features_mol, num_features_mol)
        self.mol_conv2 = GATConv(num_features_mol*2, num_features_mol * 2)
        self.mol_conv3 = SuperGATConv(num_features_mol * 4, num_features_mol * 4)
        self.mol_fc_g1 = nn.Linear(num_features_mol * 4, 256)
        self.mol_fc_g2 = nn.Linear(256, output_dim)

        # 蛋白质图分支
        self.pro_conv1 = GCNConv(num_features_pro, num_features_pro)
        self.pro_conv2 = GATConv(num_features_pro*2, num_features_pro * 2)
        self.pro_conv3 = SuperGATConv(num_features_pro * 4, num_features_pro * 4)
        self.pro_fc_g1 = nn.Linear(num_features_pro * 4, 256)
        self.pro_fc_g2 = nn.Linear(256, output_dim)

        # 序列编码分支
        self.sm = SimpleNet(embed=True, num_features=1024, output_dim=output_dim)
        self.se = SimpleNet(embed=True, num_features=1024, output_dim=output_dim)
        self.q = SimpleNet(num_features=1024, output_dim=output_dim)
        self.d = SimpleNet(num_features=1024, output_dim=output_dim)
        
        # 指纹特征变换
        self.f1 = nn.Linear(1024, output_dim)  # e3fp
        self.f2 = nn.Linear(315, output_dim)   # ergfp
        self.f3 = nn.Linear(881, output_dim)   # pubfp
        self.f4 = nn.Linear(166, output_dim)   # maccsfp
        
        # 多模态融合模块
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
            output_dim=output_dim * 2,  # 输出更大的特征维度以保留更多信息
            num_heads=4,
        )
        
        # 预测头
        self.fc1 = nn.Linear(output_dim * 2, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.out = nn.Linear(512, n_output)
        
        # 正则化和激活函数
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.bn1 = nn.BatchNorm1d(1024)
        self.bn2 = nn.BatchNorm1d(512)
        self.bn_out = nn.BatchNorm1d(n_output)
    
    def forward(self, smiles, sequence, e3fp, ergfp, pubfp, maccsfp, data_mol, data_pro, smi_m, seq_m):
        # 分子图处理
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

        # 蛋白质图处理
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

        # 序列特征提取
        smiles_seq_feat = self.sm(smiles)
        protein_seq_feat = self.se(sequence)
        smi_m_feat = self.q(smi_m)
        seq_m_feat = self.d(seq_m)
        
        # 指纹特征变换
        e3fp_feat = self.f1(e3fp)
        ergfp_feat = self.f2(ergfp)
        pubfp_feat = self.f3(pubfp)
        maccsfp_feat = self.f4(maccsfp)
        
        # 多模态融合
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
        
        # 预测
        xc = self.bn1(self.fc1(fused_features))
        xc = self.relu(xc)
        xc = self.dropout(xc)
        
        xc = self.bn2(self.fc2(xc))
        xc = self.relu(xc)
        xc = self.dropout(xc)
        
        out = self.out(xc)
        return self.bn_out(out)