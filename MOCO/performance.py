#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import builtins
import math
import os
import random
import time
import warnings
import pickle
import glob
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp


from  utils import *
'''
import sys
os.chdir("/mnt/hpc/home/shilei/RongfangNie/project/Nierongfang/Moco/code/Fugue/utils")
from builder import *
from dataloader import *
from densenet import *
from embedding_extractor_test import *
from preprocessing import *
from utils import *
from parse import *
import densenet as models
import dataloader_test as dataL_test
'''
from torch.cuda.amp import autocast, GradScaler

os.chdir("/mnt/hpc/home/shilei/RongfangNie/project/Nierongfang/Moco/code/Fugue/utils")
from builder import *
from parse import *


def load_para(save_path):
    checkpoint = torch.load(save_path)
    #载入模型参数
    state_dict = checkpoint['state_dict']
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k[7:]  # remove 'module.' of dataparallel
        if name.endswith(".weight") or name.endswith(".bias"):
            if name.split(".")[2] in ["0","2"]:
                name = f'{name.split(".")[0]}.{name.split(".")[1]}.{name.split(".")[3]}'
        new_state_dict[name] = v
    return new_state_dict



'''
data,label = test_dataset[0]

frature1 = model.encoder_q(data)
frature2 = model.encoder_q(data)
frature1 == frature2
'''
#------------------------------------------------------------------
#umap 绘制
import matplotlib.pyplot as plt
import umap

def self_umap(X, Y, pdf_dir):
    # 首先，创建一个颜色映射，每个唯一标签一个颜色
    unique_labels = np.unique(Y)
    # 创建一个颜色映射，使用 'viridis' 颜色映射
    color_map = plt.cm.get_cmap('viridis')
    # 将每个唯一标签映射到颜色映射中的一个颜色
    colors = [color_map(i / (len(unique_labels )- 1)) for i in range(len(unique_labels))]
    # 为每个标签分配一个颜色
    label_to_color = dict(zip(unique_labels, colors))
    # 使用映射后的标签颜色绘制散点图
    color_indices = np.array([label_to_color[label] for label in Y])
    # 创建UMAP对象
    umap_model = umap.UMAP(n_neighbors=10, min_dist=0.1, n_components=2,random_state=2023)
    # 将高维数据映射到低维空间
    umap_result = umap_model.fit_transform(X)
    # 打印结果
    print(umap_result)
    #plt.scatter(umap_result[:, 0], umap_result[:, 1], c=color_indices, cmap='viridis')
    # 绘制散点图，为每种肿瘤类型分配颜色
    for i, tumor_type in enumerate(unique_labels):
        mask = Y == tumor_type
        plt.scatter(umap_result[mask, 0], umap_result[mask, 1], 
                c=colors[i], label=tumor_type)
    # 添加图例
    plt.legend(loc='best')  # 你可以根据需要调整图例的位置
    plt.title('UMAP (Moco feature)')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.savefig(f"{pdf_dir}",bbox_inches='tight',dpi=300)
    plt.close()


def model_out(data,model):
    X = []
    Y = []
    labels = data["y"]
    for i in range(len(test_dataset)):
        data= test_dataset[i]
        q = torch.tensor(data[0],dtype=torch.float32)
        q = q.unsqueeze(0)
        #frature = q.detach().numpy()[0]
        frature = model.encoder_q(q).detach().numpy()[0]
        label = labels[i]
        if label.split(".")[0] in labe_data["Patient_ID"].values:
            new_label = labe_data[labe_data["Patient_ID"] == label.split(".")[0]]["Cancer"].values[0]
        X.append(frature)
        Y.append(new_label)
    X = np.array(X)
    Y = np.array(Y)
    return X,Y



#---------------------------------------------------------------------------
args = moco_parser().parse_args()

model_dict = models.__dict__

kwargs = {"randomzero" : 0.3}
model = MoCo(
        model_dict["densenet21"],
        num_batches = 8, 
        in_features = 10000, 
        dim=256, 
        K=128, 
        m=args.moco_m, 
        T=args.moco_t, 
        mlp=args.mlp,
        **kwargs)
#---------------------------------------------------------------------------------------------------------------------
'''
    CPTAC
'''
#载入数据
#data_dir = "/mnt/hpc/home/shilei/RongfangNie/project/Nierongfang/Moco/input/zscore/test/CPTAC/zscore_test_cptac.csv"
data_dir = "/mnt/hpc/home/shilei/RongfangNie/project/Nierongfang/Moco/code/Fugue/data/zscore_train_cptac.npz"
#in_dir = "/mnt/hpc/home/shilei/RongfangNie/data/GDSC/EXP/Cell_line_RMA_proc_basalExp.txt"
    # 读取数据集
test_dataset = load_data(file = data_dir, shuffle_ratio = args.shuffle_ratio, 
            load_split_file = args.load_split_file, split_now = args.split_now,split_savedir = args.split_savedir)
len(test_dataset)


# 读取标签数据
label_dir = "/mnt/hpc/home/shilei/RongfangNie/project/Nierongfang/Moco/input/zscore/test/CPTAC/clinical_test_cptac.csv"
labe_data = pd.read_csv(label_dir)
data = np.load(data_dir,allow_pickle = True)


#载入checkpoint文件
#save_path = "/mnt/hpc/home/shilei/RongfangNie/project/Nierongfang/TumorDrugPredict/code/moco-main copy 2/model/checkpoint_0041.pth copy.tar"
save_path = "/mnt/hpc/home/shilei/RongfangNie/project/Nierongfang/Moco/code/Fugue/result/checkpoint_0029.pth.tar"
model.load_state_dict(load_para(save_path))

model.eval()


X,Y = model_out(data,model)
pdf_dir = "/mnt/hpc/home/shilei/RongfangNie/project/Nierongfang/Moco/code/Fugue/8Moco_CPTAC_umap_train_0029.pdf"
self_umap(X, Y, pdf_dir)

# 原始数据
test_data_p = pd.read_csv(data_dir,index_col=0)
X1 = []
Y1=[]
for i in range(len(test_data_p)):
    frature = test_data_p.iloc[i,:].values.tolist()
    label =test_data_p.iloc[i,:].name
    if label.split(".")[0] in labe_data["Patient_ID"].values:
        new_label1 = labe_data[labe_data["Patient_ID"] == label.split(".")[0]]["Cancer"].values[0]
        X1.append(frature)
        Y1.append(new_label1)

X1 = np.array(X1)
Y1 = np.array(Y1)
pdf_dir1 = "/mnt/hpc/home/shilei/RongfangNie/project/Nierongfang/Moco/outpput/pre_CPTAC_umap_train.pdf"
self_umap(X1, Y1, pdf_dir1)

#---------------------------------------------------------------------------------------------------------------------
'''
    TCGA
'''
data_dir = "/mnt/hpc/home/shilei/RongfangNie/project/Nierongfang/Moco/input/zscore/test/zscore_test_Pan_clean.npz"
    # 读取数据集
test_dataset = load_data(file = data_dir, shuffle_ratio = args.shuffle_ratio, 
            load_split_file = args.load_split_file, split_now = args.split_now,split_savedir = args.split_savedir)
import pandas as pd
import numpy as np
na_data = np.load(data_dir,allow_pickle = True)
label_dir = "/mnt/hpc/home/shilei/RongfangNie/data/TCGA/Survival_SupplementalTable_S1_20171025_xena_sp"
labe_data = pd.read_table(label_dir,sep="\t")

X = []
Y= []
for i in range(len(test_dataset)):
    data1,data2,_ = test_dataset[i]
    label = na_data["y"][i]
    q = torch.tensor(data1,dtype=torch.float32)
    q = q.unsqueeze(0)
    frature = model.encoder_q(q).detach().numpy()[0]
    if label in labe_data["sample"].values:
        new_label = labe_data[labe_data["sample"] == label]["cancer type abbreviation"].values[0]
    X.append(frature)
    Y.append(new_label)

X = np.array(X)
Y = np.array(Y)
pdf_dir = "/mnt/hpc/home/shilei/RongfangNie/project/Nierongfang/Moco/code/Fugue/Moco_TCGA_umap_019.pdf"
self_umap(X, Y, pdf_dir)