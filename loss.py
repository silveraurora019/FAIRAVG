# 文件名: new_loss.py
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple

# 从 FedU 导入
from client_regularizers import UncertaintyTax


class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0, activation='sigmoid'):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.activation = activation

    def dice_coef(self, pred, gt):
        """计算 Dice 系数 (用于评估，非损失)"""
        softmax_pred = torch.nn.functional.softmax(pred, dim=1)
        seg_pred = torch.argmax(softmax_pred, dim=1)
        all_dice = 0
        
        if gt.dim() == 4:
            gt = gt.squeeze(dim=1) # (B, 1, H, W) -> (B, H, W)
            
        if gt.dim() != 3:
             # 确保 gt 是 (B, H, W)
             return 0.0

        batch_size = gt.shape[0]
        num_class = softmax_pred.shape[1]

        # 遍历所有类别 (包括背景)
        for i in range(num_class):
            each_pred = torch.zeros_like(seg_pred)
            each_pred[seg_pred==i] = 1

            each_gt = torch.zeros_like(gt)
            each_gt[gt==i] = 1            

        
            intersection = torch.sum((each_pred * each_gt).view(batch_size, -1), dim=1)
            
            union = each_pred.view(batch_size,-1).sum(1) + each_gt.view(batch_size,-1).sum(1)
            # 添加平滑项 eps
            dice = (2. * intersection + 1e-5) / (union + 1e-5)
         
            all_dice += torch.mean(dice)
 
        return all_dice * 1.0 / num_class


    def forward(self, pred, gt):
        """计算 Dice 损失"""
        sigmoid_pred = F.softmax(pred,dim=1)

        batch_size = gt.shape[0]
        num_class = sigmoid_pred.shape[1]
    
        # (B, 1, H, W) -> (B, H, W)
        if gt.dim() == 4:
            gt = gt.squeeze(dim=1)
            
        # 创建 one-hot 编码的 gt
        # 假设 gt 是 (B, H, W) 且类别是 0, 1, ... num_class-1
        label_one_hot = F.one_hot(gt.long(), num_classes=num_class).permute(0, 3, 1, 2) # (B, H, W, C) -> (B, C, H, W)

        loss = 0
        smooth = 1e-5

        for i in range(num_class):
            intersect = torch.sum(sigmoid_pred[:, i, ...] * label_one_hot[:, i, ...])
            z_sum = torch.sum(sigmoid_pred[:, i, ...] )
            y_sum = torch.sum(label_one_hot[:, i, ...] )
            loss += (2 * intersect + smooth) / (z_sum + y_sum + smooth)
            
        # 平均 Dice 系数，然后 1 - Dice
        loss = 1 - loss / num_class
        return loss

class JointLoss(nn.Module):
    def __init__(self, n_classes=2):
        super(JointLoss, self).__init__()
        # CrossEntropyLoss 期望 gt 为 (B, H, W) 且类型为 Long
        self.ce = nn.CrossEntropyLoss()
        self.dice = DiceLoss()
        self.n_classes = n_classes

    def forward(self, pred, gt):
        # 确保 gt 格式正确
        if gt.dim() == 4:
             gt = gt.squeeze(axis=1) # (B, 1, H, W) -> (B, H, W)
             
        # 检查 gt 是否包含超出范围的类别
        if gt.max() >= self.n_classes:
             gt = torch.clamp(gt, max=self.n_classes - 1)
             
        ce_loss = self.ce(pred, gt.long())
        dice_loss = self.dice(pred, gt)
        
        return (ce_loss + dice_loss) / 2


        
        return total_loss, task_loss.detach(), reg_loss.detach(), U_i_batch.detach()
