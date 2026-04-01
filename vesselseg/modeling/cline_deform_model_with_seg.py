import logging
import torch
from detectron2.modeling import META_ARCH_REGISTRY, build_backbone
from torch import nn
from .layers.structures import ImageList3d
from .layers.coords import batched_dist_map
from .layers.conv_blocks import ShapeSpec3d, get_dice_coeff
from .cline_deform_with_seg import Cline_Deformer

# 【新增】导入 Soft-clDice Loss
# 确保你已经按照上一步创建了 vesselseg/loss_cldice.py 文件
from vesselseg.loss_cldice import soft_cldice, tversky_loss

__all__ = ["ClineDeformModel", ]

logger = logging.getLogger(__name__)

@META_ARCH_REGISTRY.register()
class ClineDeformModel(nn.Module):

    def __init__(self, cfg):
        print(cfg)
        super(ClineDeformModel, self).__init__()
        self.backbone = build_backbone(cfg, ShapeSpec3d(channels=1))
        self.cline_deformer = Cline_Deformer(cfg)
        self.feats = ["feat_4x"]
        self.merger = nn.Conv3d(in_channels=2, out_channels=1, kernel_size=(3,3,3), stride=1, padding=1)
        self.scratcher = nn.Conv3d(in_channels=1, out_channels=1, kernel_size=(3,3,3), stride=1, padding=1)
        self.loss_type = cfg.MODEL.LOSS
        
        # 【新增】初始化 Soft-clDice Loss
        # iter_=3 是推荐值，能提取比较鲁棒的骨架
        self.cldice_loss_func = soft_cldice(iter_=3)
        
    @property
    def device(self):
        return list(self.parameters())[0].device

    def forward(self, batched_inputs):
        if not self.training:
            return self.inference(batched_inputs)
        
        assert len(batched_inputs) == 1

        images = self.preprocess_image(batched_inputs, "image")
        gt_seg = self.preprocess_image(batched_inputs, "seg")
        # 确保 GT 是 0/1 二值
        gt_seg = gt_seg.tensor.gt(0.5).float()
        pad_img_shape = images.tensor.shape

        backbone_outputs = self.backbone(images.tensor)
        features = [backbone_outputs[key] for key in self.feats]
    
        losses = dict()
        pred_seg_ = backbone_outputs["seg"] # 原始分割特征 (Batch, Channel, D, H, W)
        pred_seg_s = self.scratcher(pred_seg_)
        loss_dice_ = 1 - get_dice_coeff(pred_seg_s[:, 0].sigmoid(), gt_seg[:, 0])
        
        # 1. 调用变形器获得预测骨架
        pred_cline, loss_cline = self.cline_deformer(batched_inputs, features, pad_img_shape, 
                                                     gt_seg,  pred_seg_s.sigmoid()) 
        losses.update(loss_cline)
        
        cline_label = batched_inputs[0]['cline'][0]
        
        # 训练稳定性策略
        use_pred = (torch.rand(1) > 0.35 and loss_cline['loss_local_chamfer_2'] < 0.05)
        
        if use_pred:
            gt_cpoints = pred_cline['pred_cline']['verts']
        else:
            gt_cpoints = torch.nonzero(cline_label)

        # 2. 生成距离图 (Distance Map)
        dist_map = batched_dist_map(pred_seg_[0][0].shape, [gt_cpoints.to(self.device)])
        
        # =================================================================
        # 【骨架引导的空间注意力 (Skeleton-Guided Spatial Attention)】
        # =================================================================
        spatial_attention = 1.0 + torch.exp(-0.5 * dist_map)
        refined_features = pred_seg_ * spatial_attention
        # =================================================================
        
        # 3. 显式位置提示 (Mask Hint)
        dist_map_hint = torch.clip(dist_map, min=0, max=3).int()
        dist_map_hint = 1. - dist_map_hint.float() / 3.
        
        # 4. 最终拼接与预测
        pred_seg = torch.cat([refined_features, dist_map_hint], dim=1)
        pred_seg = self.merger(pred_seg)
        
        # =================================================================
        # 【核心修改：计算 Tversky Loss + clDice 联合 Loss】
        # 目标：解决血管预测偏细问题，提升 Dice
        # =================================================================
        
        # A. 计算 Tversky Loss (替代原 Dice Loss)
        # 记得确保文件开头有: from vesselseg.loss_cldice import soft_cldice, tversky_loss
        probs = pred_seg[:, 0].sigmoid()
        
        # alpha=0.3, beta=0.7: 强迫模型更多地召回血管 (解决 Under-segmentation)
        loss_tversky = tversky_loss(probs, gt_seg[:, 0], alpha=0.3, beta=0.7)
        
        # B. 计算 clDice Loss (拓扑约束)
        probs_5d = probs.unsqueeze(1)
        gt_5d = gt_seg[:, 0].unsqueeze(1)
        
        loss_cldice_val = self.cldice_loss_func(gt_5d, probs_5d)
        
        # C. 联合 Loss
        # 0.6 * Tversky + 0.4 * clDice
        total_loss = 0.7 * loss_tversky + 0.3 * loss_cldice_val
        
        # =================================================================
        
        # 更新 Loss 字典
        losses.update(dict(loss_dice_scratch = loss_dice_ * 0.2,
                           loss_dice=total_loss)) # 用 total_loss 作为主优化目标

        return losses

    def inference(self, batched_inputs):
        assert len(batched_inputs) == 1
        
        # 1. 预处理
        images = self.preprocess_image(batched_inputs, "image")
        image_sizes = images.image_sizes
        pad_img_shape = images.tensor.shape
        
        # === 定义单次前向传播的内部函数 ===
        def run_forward(img_tensor):
            # Backbone
            backbone_outputs = self.backbone(img_tensor)
            features = [backbone_outputs[key] for key in self.feats]
            pred_seg_ = backbone_outputs["seg"]
            pred_seg_s = self.scratcher(pred_seg_)
            
            # Deformer (骨架提取)
            # 注意：这里我们传入空的 gt_seg，因为推理时没有 GT
            ret = self.cline_deformer(batched_inputs, features, img_tensor.shape, [], pred_seg_s)
            pred_cpoints = ret['pred_cline']['verts']
            
            # Distance Map & Attention (你的核心改进)
            # 确保不传 radii
            dist_map = batched_dist_map(pred_seg_[0][0].shape, [pred_cpoints.to(self.device)])
            spatial_attention = 1.0 + torch.exp(-0.5 * dist_map)
            refined_features = pred_seg_ * spatial_attention
            
            # Hint & Merge
            dist_map_hint = torch.clip(dist_map, min=0, max=3).int()
            dist_map_hint = 1. - dist_map_hint.float() / 3.
            pred_seg = torch.cat([refined_features, dist_map_hint], dim=1)
            pred_seg = self.merger(pred_seg)
            
            return pred_seg, ret

        # === 核心逻辑：TTA (Test-Time Augmentation) ===
        
        # A. 正常预测
        pred_logits_1, ret_1 = run_forward(images.tensor)
        
        # B. 翻转预测 (水平翻转 - W维度)
        # 把左肾翻转成右肾的样子，利用模型对右侧更好的泛化能力
        img_flipped = torch.flip(images.tensor, dims=[-1]) 
        pred_logits_2_flipped, _ = run_forward(img_flipped)
        
        # 把结果翻转回来
        pred_logits_2 = torch.flip(pred_logits_2_flipped, dims=[-1])
        
        # C. 结果平均融合
        pred_logits_avg = (pred_logits_1 + pred_logits_2) / 2.0
        
        # 后处理
        pred_seg = pred_logits_avg.sigmoid().gt(0.5)
        pred_seg = pred_seg[0, 0, :image_sizes[0][0], :image_sizes[0][1], :image_sizes[0][2]]

        output = dict(seg=pred_seg)
        # 返回第一次的骨架信息用于可视化（骨架没法简单平均，影响不大）
        output.update(ret_1) 

        return [output]

    def preprocess_image(self, batched_inputs, key="image"):
        """
        Normalize, pad and batch the input images.
        """
        images = [x[key].to(self.device) for x in batched_inputs]
        if self.training:
            images = ImageList3d.from_tensors(images, (16, 16, 16))
        else:
            images = ImageList3d.from_tensors(images, (16, 16, 16))
        return images