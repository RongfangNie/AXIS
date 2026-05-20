# %%
import pandas as pd 

# %%

drug_f_dir = '../../data/DrugComb/Process/cid_features_Roberta.pkl'
kegg_dir = "../../data/DrugComb/Process/DepMap_kegg.csv"
gene_dir = "../../data/DrugComb/Process/MOCO_feature_Depmap_ProteinCodingGene.pkl"
protein_dir = "../../data/DrugComb/Process/TCPA_CCLE_RPPA500.tsv"
meta_dir = "../../data/DrugComb/Process/Metabolomics_subsetted.csv"
model_dir = "../../data/DrugComb/Process/Model.csv"
geneEffect_dir = "../../data/DrugComb/Process/MOCO_feature_Depmap_geneEffect.pkl"
ssGSEA_dir = "../../data/DrugComb/Process/MOCO_feature_Depmap_ssGSEA.pkl"
geneDependency_dir = "../../data/DrugComb/Process/MOCO_feature_Depmap_geneDependency.pkl"
methylation_dir = "../../data/DrugComb/Process/MOCO_feature_Depmap_methylation.pkl"
CNV_dir = "../../data/DrugComb/Process/MOCO_feature_Depmap_CNV.pkl"
mutation_dir = "../../data/DrugComb/Process/Depmap/Depmap_mutation_top_var_256.pkl"

cids_dir = "../../data/DrugComb/Process/DrugComb_all_witch_CID_data_name2_CID_dict.pkl"


kegg = pd.read_csv(kegg_dir,index_col=0,header=0)
protein = pd.read_table(protein_dir,sep='\t',index_col=0,header=0)
gene = pd.read_pickle(gene_dir)
pc_ls = [i.split("_")[0] for i in protein.index.to_list()]
protein.index = pc_ls


meta = pd.read_csv(meta_dir,index_col=0)
cell_model = pd.read_csv(model_dir,index_col=0)
ssGSEA = pd.read_pickle(ssGSEA_dir)
geneEffect = pd.read_pickle(geneEffect_dir)
geneDependency = pd.read_pickle(geneDependency_dir)
methylation = pd.read_pickle(methylation_dir)
CNV = pd.read_pickle(CNV_dir)
mutation = pd.read_pickle(mutation_dir)

# 定义替换规则
index_replacement_rules = {
    'COLO699': 'CHL1DM',
    'D341MED': 'D341Med'
}

protein = protein.rename(index=index_replacement_rules)


# %%
drug_cids = pd.read_pickle(cids_dir)
drug_f = pd.read_pickle(drug_f_dir)
geneT = gene.T
metaT = meta.T
proteinT = protein.T
keggT = kegg.T
geneEffectT = geneEffect.T
ssGSEAT = ssGSEA.T
geneDependencyT = geneDependency.T
methylationT = methylation.T
CNVT = CNV.T
mutationT = mutation.T

model_cell_name_ls = cell_model["StrippedCellLineName"].unique().tolist()
import torch


no_in_gene = []
no_in_meta = []
no_in_kegg = []
no_in_protein = []
no_in_geneEffect = []
no_in_ssGSEA = []
no_in_geneDependency = []
no_in_methylation = []
no_in_CNV = []
no_in_mutation = []
def pro_data(drug_row,drug_col,cell_line_name):
    data = {}
    data["drug_row"] = drug_row
    data["drug_col"] = drug_col
    data["cell_line_name"] = cell_line_name
    data["drug_f1"] = drug_f[drug_cids[drug_row]]
    data["drug_f2"] = drug_f[drug_cids[drug_col]]
    cell_name = cell_line_name
    if cell_name in model_cell_name_ls:
        cl_id = cell_model[cell_model["StrippedCellLineName"]==cell_name].index.tolist()[0]
    else:
        cl_id = None
    if cl_id in metaT.columns.tolist():
        data["meta_f"] = metaT[cl_id]
    else:
        no_in_meta.append(cell_name)
        data["meta_f"] = metaT.mean(axis=1)
    if cl_id in kegg.columns.tolist():
        data["kegg_f"] = kegg[cl_id]
    else:
        no_in_kegg.append(cell_name)
        data["kegg_f"] = kegg.mean(axis=1)
    if cl_id in geneT.columns.tolist():
        data["gene_f"] = geneT[cl_id]
    else:
        no_in_gene.append(cell_name)
        data["gene_f"] = geneT.mean(axis=1)
    
    if cell_name in proteinT.columns.tolist():
        data["protein_f"] = proteinT[cell_name]
    else:
        no_in_protein.append(cell_name)
        data["protein_f"] = proteinT.mean(axis=1)
    if cl_id in geneEffectT.columns.tolist():
        data["geneEffect_f"] = geneEffectT[cl_id]
    else:
        no_in_geneEffect.append(cell_name)
        data["geneEffect_f"] = geneEffectT.mean(axis=1)
    if cl_id in ssGSEAT.columns.tolist():
        data["ssGSEA_f"] = ssGSEAT[cl_id]
    else:
        no_in_ssGSEA.append(cell_name)
        data["ssGSEA_f"] = ssGSEAT.mean(axis=1)
    if cl_id in geneDependencyT.columns.tolist():
        data["geneDependency_f"] = geneDependencyT[cl_id]
    else:
        no_in_geneDependency.append(cell_name)
        data["geneDependency_f"] = geneDependencyT.mean(axis=1)
    if cl_id in methylationT.columns.tolist():
        data["methylation_f"] = methylationT[cl_id]
    else:
        no_in_methylation.append(cell_name)
        data["methylation_f"] = methylationT.mean(axis=1)
    if cl_id in CNVT.columns.tolist():
        data["CNV_f"] = CNVT[cl_id]
    else:
        no_in_CNV.append(cell_name)
        data["CNV_f"] = CNVT.mean(axis=1)
    if cl_id in mutationT.columns.tolist():
        data["mutation_f"] = mutationT[cl_id]
    else:
        no_in_mutation.append(cell_name)
        data["mutation_f"] = mutationT.mean(axis=1)
    return data

# %%
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader



# 2. 优化 Dataset 类
class mydata1(Dataset):
    def __init__(self, data):
        self.data = data
        # 预处理数据
        self._preprocess_data()
        
    def _preprocess_data(self):
        # 提前将数据转换为张量并移到CPU内存
        self.processed_data = []
        for item in self.data:
            processed_item = {
                "drug_f1": torch.as_tensor(item['drug_f1'], dtype=torch.float32),
                "drug_f2": torch.as_tensor(item['drug_f2'], dtype=torch.float32),
                "molt5_f1": torch.as_tensor(item['molt5_f1'], dtype=torch.float32),
                "molt5_f2": torch.as_tensor(item['molt5_f2'], dtype=torch.float32),
                "gene_f": torch.as_tensor(item['gene_f'], dtype=torch.float32),
                "protein_f": torch.as_tensor(item['protein_f'], dtype=torch.float32),
                "kegg_f": torch.as_tensor(item['kegg_f'], dtype=torch.float32),
                "meta_f": torch.as_tensor(item['meta_f'], dtype=torch.float32),
                "zip": torch.as_tensor([item['zip']], dtype=torch.float32),
                "loewe": torch.as_tensor([item['loewe']], dtype=torch.float32),
                "hsa": torch.as_tensor([item['hsa']], dtype=torch.float32),
                "bliss": torch.as_tensor([item['bliss']], dtype=torch.float32),
                "geneEffect_f": torch.as_tensor(item['geneEffect_f'], dtype=torch.float32),
                "ssGSEA_f": torch.as_tensor(item['ssGSEA_f'], dtype=torch.float32),
                "geneDependency_f": torch.as_tensor(item['geneDependency_f'], dtype=torch.float32),
                "methylation_f": torch.as_tensor(item['methylation_f'], dtype=torch.float32),
                "CNV_f": torch.as_tensor(item['CNV_f'], dtype=torch.float32),
                "mutation_f": torch.as_tensor(item['mutation_f'], dtype=torch.float32),
                #"HSA": torch.as_tensor([item['synergy'][0]], dtype=torch.float32)
            }
            self.processed_data.append(processed_item)

    def __len__(self):
        return len(self.processed_data)
    
    def __getitem__(self, index):
        return self.processed_data[index]

# %%
from torch.utils.data import ConcatDataset
#from torch.utils.data import Dataset, DataLoader
test_fold = [0]
data_fold_ls = [0,1,2,3,4,5]
tran_fold = [i for i in data_fold_ls if i not in test_fold]
data_set_dir = "../../data/DrugComb"

train_0 = torch.load(f'{data_set_dir}/drugcomb_fold_{tran_fold[0]}_data.pt')
train_1 = torch.load(f'{data_set_dir}/drugcomb_fold_{tran_fold[1]}_data.pt')
train_2 = torch.load(f'{data_set_dir}/drugcomb_fold_{tran_fold[2]}_data.pt')
train_3 = torch.load(f'{data_set_dir}/drugcomb_fold_{tran_fold[3]}_data.pt')
test_0 = torch.load(f'{data_set_dir}/drugcomb_fold_{test_fold[0]}_data.pt')

combined_train = ConcatDataset([test_0,train_0,train_1,train_2,train_3])

# %%
del train_0, train_1, train_2, train_3, test_0

# %%
len(combined_train)

# %%
train_dataloader = DataLoader(
    combined_train, 
    batch_size=len(combined_train),
    shuffle=True)

# %%
for batch in train_dataloader:
    drug_f1 = batch["drug_f1"][:,0:768]
    drug_f2 = batch["drug_f2"][:,0:768]
    synergy_zip = batch["zip"][:,0]
    gene_f = batch["gene_f"][:,0:256]
    kegg_f = batch["kegg_f"][:,0:178]
    protein_f = batch["protein_f"][:,0:447]
    meta_f = batch["meta_f"][:,0:225]
    geneEffect_f = batch["geneEffect_f"]
    ssGSEA_f = batch["ssGSEA_f"]
    geneDependency_f = batch["geneDependency_f"]
    methylation_f = batch["methylation_f"]
    CNV_f = batch["CNV_f"]
    mutation_f = batch["mutation_f"]
    break

backgroud_data = torch.cat([drug_f1,drug_f2,gene_f,kegg_f,protein_f,meta_f,geneEffect_f,ssGSEA_f,geneDependency_f,methylation_f,CNV_f,mutation_f],
                            dim=1)

# %%
import shap
import pandas as pd

# 假设 K 已经通过收敛测试确定为 150
K = 5120

# 1. 从【完整训练集 X_train】创建背景数据。此步只执行一次！
background_data_kmean = shap.kmeans(backgroud_data, K)

# %%
import torch

torch.save(background_data_kmean, '../../data/SHAP/drugcomb_backgroud_data_shap_kmeans_K-5120_sbatch.pt')
