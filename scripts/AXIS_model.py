import numpy as np
import torch.optim as optim
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.nn.init as init

# %%
def param_init(model):
    for name, param in model.named_parameters():
        if param.requires_grad:
            if 'weight' in name and param.data.dim() == 2:
                print(name)
                #break
                nn.init.kaiming_uniform_(param)
            elif 'bias' in name:
                init.zeros_(param)
            else:
                #init.zeros_(param)
                init.constant_(param, 0.1)
                
    return model


class EmbeddingLayer(nn.Module):
    def __init__(self, feature_length,hid_dim):
        super().__init__()
        self.embedding = nn.Linear(feature_length, hid_dim)
        self.head_dim = hid_dim

    def forward(self, x):
        #print(x.shape[0])
        return self.embedding(x).view(x.shape[0],1,self.head_dim)


class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout):
        super().__init__()
        assert hid_dim % n_heads == 0
        self.hid_dim = hid_dim
        self.n_heads = n_heads
        self.head_dim = hid_dim // n_heads
        self.fc_q = nn.Linear(hid_dim, hid_dim)
        self.fc_k = nn.Linear(hid_dim, hid_dim)
        self.fc_v = nn.Linear(hid_dim, hid_dim)
        self.fc_o = nn.Linear(hid_dim, hid_dim)
        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim]))

    def forward(self, query, key, value, mask=None):
        batch_size = query.shape[0]
        # query = [batch size, query len, hid dim]
        # key = [batch size, key len, hid dim]
        # value = [batch size, value len, hid dim]
        # mask = [batch size, 1, 1, key len]
        Q = self.fc_q(query)
        K = self.fc_k(key)
        V = self.fc_v(value)
        # Q = [batch size, query len, hid dim]
        # K = [batch size, key len, hid dim]
        # V = [batch size, value len, hid dim]
        Q = Q.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        # Q = [batch size, n heads, query len, head dim]
        # K = [batch size, n heads, key len, head dim]
        # V = [batch size, n heads, value len, head dim]
        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale.to(Q.device)
        # energy = [batch size, n heads, query len, key len]
        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)
        self.attention = torch.softmax(energy, dim=-1)
        # attention = [batch size, n heads, query len, key len]
        x = torch.matmul(self.dropout(self.attention), V)
        # x = [batch size, n heads, query len, head dim]
        x = x.permute(0, 2, 1, 3).contiguous()
        # x = [batch size, query len, n heads, head dim]
        x = x.view(batch_size, -1, self.hid_dim)
        # x = [batch size, query len, hid dim]
        x = self.fc_o(x)
        # x = [batch size, query len, hid dim]
        return x


#@save
class AddNorm(nn.Module):
    """残差连接后进行层规范化"""
    def __init__(self, normalized_shape, dropout, **kwargs):
        super(AddNorm, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(normalized_shape)

    def forward(self, X, Y):
        return self.ln(self.dropout(Y) + X)

#@save
class EncoderBlock(nn.Module):
    """Transformer编码器块"""
    def __init__(self, key_size, query_size, value_size, num_hiddens,
                 norm_shape, num_heads,
                 dropout, use_bias=False, **kwargs):
        super(EncoderBlock, self).__init__(**kwargs)
        self.attention = MultiHeadAttentionLayer(
            num_hiddens, num_heads, dropout)
        self.addnorm1 = AddNorm(norm_shape, dropout)

    def forward(self, Q,K,V):
        return self.addnorm1(Q, self.attention(Q, K, V))

#@save
class TransformerEncoder(nn.Module):
    """Transformer编码器"""
    def __init__(self, feature_size, key_size, query_size, value_size,
                 num_hiddens, norm_shape,num_heads, num_layers, dropout, use_bias=False, **kwargs):
        super(TransformerEncoder, self).__init__(**kwargs)
        self.num_hiddens = num_hiddens
        self.embedding = EmbeddingLayer(feature_size, num_hiddens)
        self.blks = nn.Sequential()
        for i in range(num_layers):
            self.blks.add_module("block"+str(i),
                EncoderBlock(key_size, query_size, value_size, num_hiddens,
                             norm_shape,num_heads, dropout, use_bias))

    def forward(self, Q,K,V,*args):
        # 因为位置编码值在-1和1之间，
        # 因此嵌入值乘以嵌入维度的平方根进行缩放，
        # 然后再与位置编码相加。
        Q = self.embedding(Q)
        K = self.embedding(K)
        V = self.embedding(V)
        self.attention_weights = [None] * len(self.blks)
        for i, blk in enumerate(self.blks):
            X = blk(Q,K,V)
            self.attention_weights[
                i] = blk.attention
        return X
    
# 示例网络架构优化
class ImprovedSelfAttentionNetwork(nn.Module):
    def __init__(self, hidden_dim, num_heads):
        super().__init__()
        # 增加网络深度和复杂性
        self.attention_layers = nn.ModuleList([
            nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads,batch_first=True,dropout=0.3) 
            for _ in range(8)  # 增加注意力层数
        ])
        
         #添加残差连接和层归一化
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(8)
        ])
        
        # 增加非线性变换
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )

        self.mlps = nn.ModuleList([
           self.mlp for _ in range(8)
        ])
        
    def forward(self, Q,K,V):
        # 实现多层注意力和残差连接
        for attention, norm, mlp in zip(self.attention_layers, self.layer_norms, self.mlps):
        #for attention in self.attention_layers:
            residual = Q
            Q, _ = attention(Q, K, V)
            Q = norm(Q + residual)
            Q = mlp(Q)
        return Q


#--------------------------------------------------------------------------------
class Predictor(nn.Module):
    def __init__(self, ) -> None:
        super().__init__()
        #feature_size = input_size
        # Drug - cell -Drug cancentration
        feature_size = 768
        Drop_rate = 0.3
        self.D12C_encoder = ImprovedSelfAttentionNetwork(hidden_dim=768, num_heads=8)
        
        self.CD12_encoder = ImprovedSelfAttentionNetwork(hidden_dim=768, num_heads=8)
        
        self.D122_encoder = ImprovedSelfAttentionNetwork(hidden_dim=768, num_heads=8)

        self.CD_encoder = ImprovedSelfAttentionNetwork(hidden_dim=768, num_heads=8)

        self.DC_encoder = ImprovedSelfAttentionNetwork(hidden_dim=768, num_heads=8)

        self.DCD_encoder = ImprovedSelfAttentionNetwork(hidden_dim=768, num_heads=8)

        self.gene_p = nn.Sequential(
            nn.Linear(256, 256)
        )

        self.kegg_p = nn.Sequential(
            nn.Linear(178, 256)
        )

        self.protein_p = nn.Sequential(
            nn.Linear(447, 256)
        )

        self.meta_p = nn.Sequential(
            nn.Linear(225, 256)
        )

        self.geneEffect_p = nn.Sequential(
            nn.Linear(256, 256)
        )

        self.ssGSEA_p = nn.Sequential(
            nn.Linear(256, 256)
        )

        self.geneDependency_p = nn.Sequential(
            nn.Linear(256, 256)
        )

        self.methylation_p = nn.Sequential(
            nn.Linear(256, 256)
        )

        self.CNV_p = nn.Sequential(
            nn.Linear(256, 256)
        )

        self.mutation_p = nn.Sequential(
            nn.Linear(256, 256)
        )

        self.drug_p = nn.Sequential(
            nn.Linear(768, 768)
        )

        self.gene_f = nn.Sequential(
            nn.Linear(2048+256*2, 1024),
            nn.Linear(1024, 768)
        )
        
        
        self.bliss0 = nn.Sequential(
            nn.Linear(768*12, 2048),
            nn.Dropout(Drop_rate),
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.Dropout(Drop_rate),
            nn.ReLU(),
            nn.Linear(1024, 256),
            nn.Dropout(Drop_rate),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

        self.bliss1 = nn.Sequential(
            nn.Linear(768*12, 4068),
            nn.Dropout(Drop_rate),
            nn.ReLU(),
            nn.Linear(4068, 2048),
            nn.Dropout(Drop_rate),
            nn.ReLU(),
            nn.Linear(2048, 768),
            nn.Dropout(Drop_rate),
            nn.ReLU(),
            nn.Linear(768, 256),
            nn.Dropout(Drop_rate),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

        self.bliss2 = nn.Sequential(
            nn.Linear(768*12, 2048),
            nn.Dropout(Drop_rate),
            nn.ReLU(),
            nn.Linear(2048, 768),
            nn.Dropout(Drop_rate),
            nn.ReLU(),
            nn.Linear(768, 128),
            nn.Dropout(Drop_rate),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

        self.var = nn.Sequential(
            nn.Linear(768*12, 2048),
            nn.Dropout(Drop_rate),
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.Dropout(Drop_rate),
            nn.ReLU(),
            nn.Linear(1024, 256),
            nn.Dropout(Drop_rate),
            nn.ReLU(),
            nn.Linear(256, 1),
        )


    def forward(self,drug_f1,drug_f2,gene_f,kegg_f,protein_f,meta_f,geneEffect_f,ssGSEA_f,geneDependency_f,methylation_f,CNV_f,mutation_f,step = None):
        #Drug-cell
        #print(d1c.size())
        drug_f1 = self.drug_p(drug_f1).unsqueeze(1)
        drug_f2 = self.drug_p(drug_f2).unsqueeze(1)
        gene_f = self.gene_p(gene_f)
        kegg_f = self.kegg_p(kegg_f)
        protein_f = self.protein_p(protein_f)
        meta_f = self.meta_p(meta_f)
        geneEffect_f = self.geneEffect_p(geneEffect_f)
        ssGSEA_f = self.ssGSEA_p(ssGSEA_f)
        geneDependency_f = self.geneDependency_p(geneDependency_f)
        methylation_f = self.methylation_p(methylation_f)
        CNV_f = self.CNV_p(CNV_f)
        mutation_f = self.mutation_p(mutation_f)
        batch_size = drug_f1.size(0)
        

        gene_f1 = self.gene_f(torch.cat((gene_f,kegg_f,protein_f,meta_f,geneEffect_f,ssGSEA_f,geneDependency_f,methylation_f,CNV_f,mutation_f),dim=1)).unsqueeze(1)


        D12C_f_1 = self.D12C_encoder(drug_f1,drug_f2,gene_f1).reshape(batch_size,-1)
        D12C_f_2 = self.D12C_encoder(drug_f2,drug_f1,gene_f1).reshape(batch_size,-1)
        CD12_f_1 = self.CD12_encoder(gene_f1,drug_f1,drug_f2).reshape(batch_size,-1)
        CD12_f_2 = self.CD12_encoder(gene_f1,drug_f2,drug_f1).reshape(batch_size,-1)
        D122_f_1 = self.D122_encoder(drug_f1,drug_f2,drug_f2).reshape(batch_size,-1)
        D122_f_2 = self.D122_encoder(drug_f2,drug_f1,drug_f1).reshape(batch_size,-1)
        CD_f_1 = self.CD_encoder(gene_f1,drug_f1,drug_f1).reshape(batch_size,-1)
        CD_f_2 = self.CD_encoder(gene_f1,drug_f2,drug_f2).reshape(batch_size,-1)
        DC_f_1 = self.DC_encoder(drug_f1,gene_f1,gene_f1).reshape(batch_size,-1)
        DC_f_2 = self.DC_encoder(drug_f2,gene_f1,gene_f1).reshape(batch_size,-1)
        DCD_f_1 = self.DCD_encoder(drug_f1,gene_f1,drug_f2).reshape(batch_size,-1)
        DCD_f_2 = self.DCD_encoder(drug_f2,gene_f1,drug_f1).reshape(batch_size,-1)

        in1 = torch.cat((D12C_f_1,D12C_f_2,CD12_f_1,CD12_f_2,D122_f_1,D122_f_2,CD_f_1,CD_f_2,DC_f_1,DC_f_2,DCD_f_1,DCD_f_2),dim=1)
        bliss1 = (self.bliss0(in1) + self.bliss1(in1) + self.bliss2(in1))/3
        var = nn.functional.softplus(self.var(in1))
        #hsa = self.hsa(torch.cat((f1,f2),dim=1))
        #bliss = self.bliss(torch.cat((f1,f2),dim=1))
        return bliss1,var


class Predictor_eval(nn.Module):
    def __init__(self, ) -> None:
        super().__init__()
        #feature_size = input_size
        # Drug - cell -Drug cancentration
        feature_size = 768
        Drop_rate = 0.3
        self.D12C_encoder = ImprovedSelfAttentionNetwork(hidden_dim=768, num_heads=8)
        
        self.CD12_encoder = ImprovedSelfAttentionNetwork(hidden_dim=768, num_heads=8)
        
        self.D122_encoder = ImprovedSelfAttentionNetwork(hidden_dim=768, num_heads=8)

        self.CD_encoder = ImprovedSelfAttentionNetwork(hidden_dim=768, num_heads=8)

        self.DC_encoder = ImprovedSelfAttentionNetwork(hidden_dim=768, num_heads=8)

        self.DCD_encoder = ImprovedSelfAttentionNetwork(hidden_dim=768, num_heads=8)

        self.gene_p = nn.Sequential(
            nn.Linear(256, 256)
        )

        self.kegg_p = nn.Sequential(
            nn.Linear(178, 256)
        )

        self.protein_p = nn.Sequential(
            nn.Linear(447, 256)
        )

        self.meta_p = nn.Sequential(
            nn.Linear(225, 256)
        )

        self.geneEffect_p = nn.Sequential(
            nn.Linear(256, 256)
        )

        self.ssGSEA_p = nn.Sequential(
            nn.Linear(256, 256)
        )

        self.geneDependency_p = nn.Sequential(
            nn.Linear(256, 256)
        )

        self.methylation_p = nn.Sequential(
            nn.Linear(256, 256)
        )

        self.CNV_p = nn.Sequential(
            nn.Linear(256, 256)
        )

        self.mutation_p = nn.Sequential(
            nn.Linear(256, 256)
        )

        self.drug_p = nn.Sequential(
            nn.Linear(768, 768)
        )

        self.gene_f = nn.Sequential(
            nn.Linear(2048+256*2, 1024),
            nn.Linear(1024, 768)
        )
        
        
        self.bliss0 = nn.Sequential(
            nn.Linear(768*12, 2048),
            nn.Dropout(Drop_rate),
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.Dropout(Drop_rate),
            nn.ReLU(),
            nn.Linear(1024, 256),
            nn.Dropout(Drop_rate),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

        self.bliss1 = nn.Sequential(
            nn.Linear(768*12, 4068),
            nn.Dropout(Drop_rate),
            nn.ReLU(),
            nn.Linear(4068, 2048),
            nn.Dropout(Drop_rate),
            nn.ReLU(),
            nn.Linear(2048, 768),
            nn.Dropout(Drop_rate),
            nn.ReLU(),
            nn.Linear(768, 256),
            nn.Dropout(Drop_rate),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

        self.bliss2 = nn.Sequential(
            nn.Linear(768*12, 2048),
            nn.Dropout(Drop_rate),
            nn.ReLU(),
            nn.Linear(2048, 768),
            nn.Dropout(Drop_rate),
            nn.ReLU(),
            nn.Linear(768, 128),
            nn.Dropout(Drop_rate),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

        self.var = nn.Sequential(
            nn.Linear(768*12, 2048),
            nn.Dropout(Drop_rate),
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.Dropout(Drop_rate),
            nn.ReLU(),
            nn.Linear(1024, 256),
            nn.Dropout(Drop_rate),
            nn.ReLU(),
            nn.Linear(256, 1),
        )


    def forward(self,drug_f1,drug_f2,gene_f,kegg_f,protein_f,meta_f,geneEffect_f,ssGSEA_f,geneDependency_f,methylation_f,CNV_f,mutation_f,step = None):
        #Drug-cell
        #print(d1c.size())
        drug_f1 = self.drug_p(drug_f1).unsqueeze(1)
        drug_f2 = self.drug_p(drug_f2).unsqueeze(1)
        gene_f = self.gene_p(gene_f)
        kegg_f = self.kegg_p(kegg_f)
        protein_f = self.protein_p(protein_f)
        meta_f = self.meta_p(meta_f)
        geneEffect_f = self.geneEffect_p(geneEffect_f)
        ssGSEA_f = self.ssGSEA_p(ssGSEA_f)
        geneDependency_f = self.geneDependency_p(geneDependency_f)
        methylation_f = self.methylation_p(methylation_f)
        CNV_f = self.CNV_p(CNV_f)
        mutation_f = self.mutation_p(mutation_f)
        batch_size = drug_f1.size(0)
        

        gene_f1 = self.gene_f(torch.cat((gene_f,kegg_f,protein_f,meta_f,geneEffect_f,ssGSEA_f,geneDependency_f,methylation_f,CNV_f,mutation_f),dim=1)).unsqueeze(1)


        D12C_f_1 = self.D12C_encoder(drug_f1,drug_f2,gene_f1).reshape(batch_size,-1)
        D12C_f_2 = self.D12C_encoder(drug_f2,drug_f1,gene_f1).reshape(batch_size,-1)
        CD12_f_1 = self.CD12_encoder(gene_f1,drug_f1,drug_f2).reshape(batch_size,-1)
        CD12_f_2 = self.CD12_encoder(gene_f1,drug_f2,drug_f1).reshape(batch_size,-1)
        D122_f_1 = self.D122_encoder(drug_f1,drug_f2,drug_f2).reshape(batch_size,-1)
        D122_f_2 = self.D122_encoder(drug_f2,drug_f1,drug_f1).reshape(batch_size,-1)
        CD_f_1 = self.CD_encoder(gene_f1,drug_f1,drug_f1).reshape(batch_size,-1)
        CD_f_2 = self.CD_encoder(gene_f1,drug_f2,drug_f2).reshape(batch_size,-1)
        DC_f_1 = self.DC_encoder(drug_f1,gene_f1,gene_f1).reshape(batch_size,-1)
        DC_f_2 = self.DC_encoder(drug_f2,gene_f1,gene_f1).reshape(batch_size,-1)
        DCD_f_1 = self.DCD_encoder(drug_f1,gene_f1,drug_f2).reshape(batch_size,-1)
        DCD_f_2 = self.DCD_encoder(drug_f2,gene_f1,drug_f1).reshape(batch_size,-1)

        D12C_f_1_2 = self.D12C_encoder(drug_f2,drug_f1,gene_f1).reshape(batch_size,-1)
        D12C_f_2_2 = self.D12C_encoder(drug_f1,drug_f2,gene_f1).reshape(batch_size,-1)
        CD12_f_1_2 = self.CD12_encoder(gene_f1,drug_f2,drug_f1).reshape(batch_size,-1)
        CD12_f_2_2 = self.CD12_encoder(gene_f1,drug_f1,drug_f2).reshape(batch_size,-1)
        D122_f_1_2 = self.D122_encoder(drug_f2,drug_f1,drug_f1).reshape(batch_size,-1)
        D122_f_2_2 = self.D122_encoder(drug_f1,drug_f2,drug_f2).reshape(batch_size,-1)
        CD_f_1_2 = self.CD_encoder(gene_f1,drug_f2,drug_f2).reshape(batch_size,-1)
        CD_f_2_2 = self.CD_encoder(gene_f1,drug_f1,drug_f1).reshape(batch_size,-1)
        DC_f_1_2 = self.DC_encoder(drug_f2,gene_f1,gene_f1).reshape(batch_size,-1)
        DC_f_2_2 = self.DC_encoder(drug_f1,gene_f1,gene_f1).reshape(batch_size,-1)
        DCD_f_1_2 = self.DCD_encoder(drug_f2,gene_f1,drug_f1).reshape(batch_size,-1)
        DCD_f_2_2 = self.DCD_encoder(drug_f1,gene_f1,drug_f2).reshape(batch_size,-1)

        in1 = torch.cat((D12C_f_1,D12C_f_2,CD12_f_1,CD12_f_2,D122_f_1,D122_f_2,CD_f_1,CD_f_2,DC_f_1,DC_f_2,DCD_f_1,DCD_f_2),dim=1)
        in2 = torch.cat((D12C_f_1_2,D12C_f_2_2,CD12_f_1_2,CD12_f_2_2,D122_f_1_2,D122_f_2_2,CD_f_1_2,CD_f_2_2,DC_f_1_2,DC_f_2_2,DCD_f_1_2,DCD_f_2_2),dim=1)
        bliss1 = (self.bliss0(in1) + self.bliss1(in1) + self.bliss2(in1))/3
        var = nn.functional.softplus(self.var(in1))
        #hsa = self.hsa(torch.cat((f1,f2),dim=1))
        #bliss = self.bliss(torch.cat((f1,f2),dim=1))
        return bliss1,var