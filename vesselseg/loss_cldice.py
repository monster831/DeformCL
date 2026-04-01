import torch
import torch.nn as nn
import torch.nn.functional as F

def soft_erode(img):
    """【数学定义】软腐蚀：使用 Min-Pooling 近似形态学腐蚀，保持可微性"""
    if len(img.shape)==4:
        p1 = -F.max_pool2d(-img, (3,1), (1,1), (1,0))
        p2 = -F.max_pool2d(-img, (1,3), (1,1), (0,1))
        return torch.min(p1,p2)
    elif len(img.shape)==5:
        p1 = -F.max_pool3d(-img, (3,1,1), (1,1,1), (1,0,0))
        p2 = -F.max_pool3d(-img, (1,3,1), (1,1,1), (0,1,0))
        p3 = -F.max_pool3d(-img, (1,1,3), (1,1,1), (0,0,1))
        return torch.min(torch.min(p1, p2), p3)

def soft_dilate(img):
    """【数学定义】软膨胀：使用 Max-Pooling 近似"""
    if len(img.shape)==4:
        return F.max_pool2d(img, (3,3), (1,1), (1,1))
    elif len(img.shape)==5:
        return F.max_pool3d(img, (3,3,3), (1,1,1), (1,1,1))

def soft_open(img):
    return soft_dilate(soft_erode(img))

def soft_skel(img, iter_):
    """【论文核心公式】可微骨架提取：S = ReLU(img - open(img)) 的迭代"""
    img1 = soft_open(img)
    skel = F.relu(img-img1)
    for i in range(iter_):
        img = soft_erode(img)
        img1 = soft_open(img)
        delta = F.relu(img-img1)
        skel = skel + F.relu(delta-skel*delta)
    return skel

class soft_cldice(nn.Module):
    def __init__(self, iter_=3, smooth=1.):
        super(soft_cldice, self).__init__()
        self.iter = iter_
        self.smooth = smooth

    def forward(self, y_true, y_pred):
        # 输入形状需为 (B, C, X, Y, Z)
        skel_pred = soft_skel(y_pred, self.iter)
        skel_true = soft_skel(y_true, self.iter)
        
        tprec = (torch.sum(torch.multiply(skel_pred, y_true)[:,1:,...])+self.smooth)/ \
                (torch.sum(skel_pred[:,1:,...])+self.smooth)    
        tsens = (torch.sum(torch.multiply(skel_true, y_pred)[:,1:,...])+self.smooth)/ \
                (torch.sum(skel_true[:,1:,...])+self.smooth)    
        
        cl_dice = 1.- 2.0*(tprec*tsens)/(tprec+tsens)
        return cl_dice

def tversky_loss(pred, target, alpha=0.3, beta=0.7, smooth=1.0):
    """
    Tversky Loss: 专门解决数据不平衡和预测偏细问题
    alpha: 惩罚假阳性 (FP)
    beta:  惩罚假阴性 (FN) -> 设为 0.7 意味着我们更怕"漏标"，强迫模型标粗一点
    """
    pred = pred.view(-1)
    target = target.view(-1)
    
    # True Positive
    tp = (pred * target).sum()
    # False Positive
    fp = ((1 - target) * pred).sum()
    # False Negative
    fn = (target * (1 - pred)).sum()
    
    tversky_index = (tp + smooth) / (tp + alpha * fp + beta * fn + smooth)
    return 1 - tversky_index