import random
import numpy as np
import torch

#---------------------------------------------------#
#   设置种子
#---------------------------------------------------#
def seed_everything(seed=11):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

#---------------------------------------------------#
#   设置Dataloader的种子
#---------------------------------------------------#
def worker_init_fn(worker_id, rank, seed):
    worker_seed = rank + seed
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)

import torch.nn as nn

loss_F = nn.GaussianNLLLoss()
loss_F2 = nn.MSELoss()

def mixup_data(x1, x2, x3, x4, x5, x6, x7, x8, x9,x10,x11,x12, y, alpha=0.2):
    """执行mixup数据增强
    
    Args:
        x1-x8: 8个输入特征
        y: 标签
        alpha: Beta分布的参数
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x1.size()[0]
    index = torch.randperm(batch_size).to(x1.device)
    
    # 对所有输入特征进行mixup
    mixed_x1 = lam * x1 + (1 - lam) * x1[index]
    mixed_x2 = lam * x2 + (1 - lam) * x2[index] 
    mixed_x3 = lam * x3 + (1 - lam) * x3[index]
    mixed_x4 = lam * x4 + (1 - lam) * x4[index]
    mixed_x5 = lam * x5 + (1 - lam) * x5[index]
    mixed_x6 = lam * x6 + (1 - lam) * x6[index]
    mixed_x7 = lam * x7 + (1 - lam) * x7[index]
    mixed_x8 = lam * x8 + (1 - lam) * x8[index]
    mixed_x9 = lam * x9 + (1 - lam) * x9[index]
    mixed_x10 = lam * x10 + (1 - lam) * x10[index]
    mixed_x11 = lam * x11 + (1 - lam) * x11[index]
    mixed_x12 = lam * x12 + (1 - lam) * x12[index]

    
    y_a, y_b = y, y[index]
    return mixed_x1, mixed_x2, mixed_x3, mixed_x4, mixed_x5, mixed_x6, mixed_x7, mixed_x8, mixed_x9, mixed_x10, mixed_x11, mixed_x12, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam,var):
    """Mixup损失函数
    
    Args:
        criterion: 原始损失函数
        pred: 模型预测
        y_a, y_b: 两个标签
        lam: mixup权重
    """
    loss_a = loss_F(pred.view(-1),y_a,var.view(-1)) + torch.mean(var.view(-1)) * 0.1
    loss_b = loss_F(pred.view(-1),y_b,var.view(-1)) + torch.mean(var.view(-1)) * 0.1
    return lam * loss_a + (1 - lam) * loss_b