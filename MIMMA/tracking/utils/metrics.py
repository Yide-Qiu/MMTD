import numpy as np
import torch
import torch.nn.functional as F
import pdb
from info_nce import InfoNCE, info_nce
from tracking.utils.losses import SupConLoss

def MSE_LOSS(output, target, mask=None):

    pred_xy = output[:,:,0,:2]
    gt_xy = target[:,:,0,:2]
    # import pdb
    # pdb.set_trace()
    norm = torch.norm(pred_xy - gt_xy, p=2, dim=-1)

    mean_K = torch.mean(norm, dim=-1)
    mean_B = torch.mean(mean_K)

    return mean_B*100

def ASSO_LOSS(out_asso, mmsi_list):
    N = len(mmsi_list)
    # pdb.set_trace()
    mmsi_matrix = torch.zeros((N, N), dtype=torch.float32).to(out_asso.device)
    for i in range(N):
        for j in range(N):
            mmsi_matrix[i][j] = 1 if mmsi_list[i] == mmsi_list[j] else 0
    # pdb.set_trace()
    # 1. 找出正样本的位置
    positive_indices = (mmsi_matrix == 1).nonzero(as_tuple=False)

    # 2. 找出负样本的位置
    negative_indices = (mmsi_matrix == 0).nonzero(as_tuple=False)

    # 3. 随机采样负样本数量与正样本相同
    num_positive_samples = positive_indices.size(0)
    negative_sample_indices = negative_indices[torch.randperm(negative_indices.size(0))[:num_positive_samples]]
    # 保留的正样本和采样的负样本的二维索引
    all_indices = torch.cat((positive_indices, negative_sample_indices), dim=0)
    # 计算二维索引对应在 x_cat 中的一维索引
    # 在 mmsi_matrix 中 (i, j) 位置对应 x_cat 中的 i * N + j 行
    linear_indices = all_indices[:, 0] * N + all_indices[:, 1]

    # 根据 linear_indices 从 x_cat 中选取对应的样本
    filtered_x_cat = out_asso[linear_indices]
    # 创建标签向量
    labels = torch.zeros(linear_indices.size(0), dtype=torch.float32).to(filtered_x_cat.device)

    # 前 num_positive_samples 个设置为 1（表示正样本）
    labels[:num_positive_samples] = 1.0
    # pdb.set_trace()

    filtered_x_cat = torch.sigmoid(filtered_x_cat)
    # pdb.set_trace()
    bce = F.binary_cross_entropy(filtered_x_cat[:, 0], labels)
    return bce

def CONTRASTIVE_LOSS(output, mmsi_list):
    # 自监督模式 InfoNCE
    info_nce_loss = InfoNCE()
    return info_nce_loss(output, output)
    # # 监督模式 SupConLoss
    # # pdb.set_trace()
    # sup_con_loss = SupConLoss(temperature=0.1)
    # return sup_con_loss(output.unsqueeze(1), mmsi_list.to(output.device))

def MSE_LOSS_XYSC(output, target, mask=None):

    pred_xy = output[:,:,0,:2]
    pred_s = output[:,:,0,2:3]
    pred_c = output[:,:,0,3:4]
    gt_xy = target[:,:,0,:2]
    gt_s = target[:,:,1,:1]
    gt_c = target[:,:,2,:1]
    # pdb.set_trace()
    norm = torch.norm(pred_xy - gt_xy, p=2, dim=-1)

    mean_K = torch.mean(norm, dim=-1)
    mean_B = torch.mean(mean_K)

    norm = torch.norm(pred_s - gt_s, p=2, dim=-1)

    mean_K = torch.mean(norm, dim=-1)
    mean_B_s = torch.mean(mean_K)

    norm = torch.norm(pred_c - gt_c, p=2, dim=-1)

    mean_K = torch.mean(norm, dim=-1)
    mean_B_c = torch.mean(mean_K)

    return mean_B*100, mean_B_s*100, mean_B_c*100
