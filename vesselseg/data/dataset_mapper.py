import random
import copy
import torch
import numpy as np
import logging
from detectron2.data.transforms import Augmentation, AugmentationList, AugInput

# 确保 transform_gen 和 transform 能正确导入
from .transform_gen import RandomCrop
from .transform import FlipTransform, SwapAxesTransform, CropTransform
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# ==========================================================
# 1. 将所有辅助增强类移到最上方 (防止 NameError)
# ==========================================================

class InferCrop(Augmentation):
    def __init__(self, type, crop_size):
        super().__init__()
        self._init(locals())
        self.type = type
        self.crop_size = crop_size

    def get_transform(self, image, sem_seg):
        h, w, d = image.shape[:3]
        if sem_seg is not None:
            if random.random() < 0.2:
                pos = np.where(sem_seg >= 0)
            else:
                pos = np.where(sem_seg > 0)
        else:
            # Fallback if no seg
            pos = (np.array([h//2]), np.array([w//2]), np.array([d//2]))

        if len(pos[0]) == 0:
            idx = 0
            pos = (np.array([h//2]), np.array([w//2]), np.array([d//2]))
        else:
            idx = np.random.randint(0, len(pos[0]))

        croph, cropw, cropd = self.crop_size if self.type == "absolute" else \
            (
                np.random.randint(int(h*self.crop_size[0]), h + 1), 
                np.random.randint(int(w*self.crop_size[1]), w + 1),
                np.random.randint(int(d*self.crop_size[2]), d + 1)
            )
        jitter = np.random.randint(-50, 50, 3)
        
        # 安全检查，防止 pos 索引溢出
        h0 = pos[0][idx] - croph // 2 + jitter[0]
        w0 = pos[1][idx] - cropw // 2 + jitter[1]
        d0 = pos[2][idx] - cropd // 2 + jitter[2]
        
        h0 = max(0, min(h - croph, h0))
        w0 = max(0, min(w - cropw, w0))
        d0 = max(0, min(d - cropd, d0))
        return CropTransform(h0, w0, d0, croph, cropw, cropd)

class RandomFlip_Z(Augmentation):
    def __init__(self, prob=0.5):
        super().__init__()
        self._init(locals())
        self.prob = prob
    def get_transform(self, img):
        flip_y = self._rand_range() < 0. # False
        flip_x = self._rand_range() < 0. # False
        flip_z = self._rand_range() < self.prob
        return FlipTransform(flip_y, flip_x, flip_z)
    
class RandomFlip_X(Augmentation):
    def __init__(self, prob=0.5):
        super().__init__()
        self._init(locals())
        self.prob = prob
    def get_transform(self, img):
        flip_y = self._rand_range() < 0.
        flip_x = self._rand_range() < self.prob
        flip_z = self._rand_range() < 0.
        return FlipTransform(flip_y, flip_x, flip_z)   

class RandomSwapAxesXZ(Augmentation):
    def __init__(self):
        super().__init__()
        self._init(locals())
    def get_transform(self, img):
        if np.random.rand() < 0.8:
            axes = [0, 1, 2]
        else:
            axes = [0, 2, 1]
        return SwapAxesTransform(axes)


# ==========================================================
# 2. 辅助函数
# ==========================================================

def interpolate(img, type='img'):
    # 防止 numpy 报 warning
    img = torch.tensor(img.copy()).unsqueeze(0).unsqueeze(0)
    if type == 'img':
        H, W, D = img.shape[2:]
        new_shape = (H, W // 2, D // 2)
        img_new = F.interpolate(img, size=new_shape, mode='trilinear', align_corners=False)
    else:
        H, W, D = img.shape[2:]
        new_shape = (H, W // 2, D // 2)
        img_new = F.interpolate(img.float(), size=new_shape, mode='nearest')
        img_new = img_new.int()
    return img_new[0, 0].numpy()

def build_cline_deform_transform_gen(cfg, is_train):
    tfm_gens = []
    crop_size = cfg.INPUT.CROP_SIZE_TRAIN
    if is_train:
        # 1. 换成精准制导裁切，避免 NaN 报错
        tfm_gens.append(InferCrop("absolute", crop_size))
        tfm_gens.append(RandomCrop("relative_range", (0.9, 0.9, 0.9)))
        # 2. 删除了随机翻转，保护门静脉真实非对称拓扑结构
        tfm_gens.append(RandomSwapAxesXZ())
    return tfm_gens

def build_bbox_transform_gen(cfg, is_train):
    tfm_gens = []
    crop_size = cfg.INPUT.CROP_SIZE_TRAIN
    if is_train:
        tfm_gens.append(InferCrop("relative_range", (0.75, 0.75, 0.75)))
        tfm_gens.append(InferCrop("absolute", crop_size))
    return tfm_gens


# ==========================================================
# 3. DatasetMapper 类定义
# ==========================================================

class ClineDeformDatasetMapper:

    def __init__(self, cfg, transform_builder, is_train=True):
        self.cfg = cfg
        self.is_train = is_train
        augmentations = transform_builder(cfg, is_train)
        self.augmentations = AugmentationList(augmentations)
        mode = "training" if is_train else "inference"
        logger.info(f"[DatasetMapper] Augmentations used in {mode}: {augmentations}")
        self.default_class_id = cfg.MODEL.PRED_CLASS

    def __call__(self, dataset_dict):
        dataset_dict = copy.deepcopy(dataset_dict)
        file_name = dataset_dict["file_name"]
        npz_file = np.load(file_name, allow_pickle=True)

        # ------------------------------------------------
        # 【修改 1】彻底移除左右侧判断，统一设为类别 1 (主动脉)
        # ------------------------------------------------
        target_class_id = 1

        pad = np.array([50, 50, 50])
        
        # 读取 cline 并转置维度 (X,Y,Z) -> (Z,Y,X)
        cline = npz_file["cline"].transpose(2, 1, 0)
        
        src_shape = cline.shape
        
        indices = np.array(np.where(cline == target_class_id))
        
        if indices.shape[1] == 0:
             start = [0, 0, 0]
             end = src_shape
        else:
             start = np.maximum(indices.min(1) - pad, 0).tolist()
             end = np.minimum(indices.max(1) + 1 + pad, src_shape).tolist()

        # ------------------------------------------------
        # 测试时 BBox 加载 (修复版)
        # ------------------------------------------------
        if not self.is_train:
            try:
                # 动态选择 bbox 文件
                bbox_path = f'bbox_pred{target_class_id}/bbox_pred.npz'
                self.pred_bbox = np.load(bbox_path, allow_pickle=True)
                
                item = self.pred_bbox['metrics'].item()
                if dataset_dict["file_id"] in item:
                    bbox = item[dataset_dict["file_id"]]['bbox_pred']['bbox_pred']
                    
                    # 计算裁切范围
                    start = (bbox[0], bbox[2] * 2, bbox[4] * 2)
                    start = np.maximum(start - pad, 0).tolist()
                    end = (bbox[1], bbox[3] * 2, bbox[5] * 2)
                    end = np.minimum(end + pad, src_shape).tolist()
                else:
                    logger.warning(f"File ID {dataset_dict['file_id']} not found in {bbox_path}")
            except Exception as e:
                # 降级为全图，防止报错
                start = [0, 0, 0]
                end = src_shape

        # ------------------------------------------------
        # 读取数据与归一化
        # ------------------------------------------------
        # 读取 image 并转置
        image = npz_file["img"].astype(np.float32).transpose(2, 1, 0)
        image = image[start[0]:end[0], start[1]:end[1], start[2]:end[2]]
        
        # 读取 seg 并转置
        seg = npz_file["seg"].transpose(2, 1, 0)
        seg = seg[start[0]:end[0], start[1]:end[1], start[2]:end[2]]

        # 标签统一化 (都变 1)
        seg_binary = np.zeros_like(seg)
        seg_binary[seg == target_class_id] = 1 
        seg = seg_binary 

        # 【修改 2】彻底移除了 np.flip() 左右翻转逻辑，保持主动脉的真实空间位置

        aug_input = AugInput(image=image, sem_seg=seg)
        transforms = self.augmentations(aug_input)
        image = aug_input.image
        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image[None, ...]))

        cline = cline[start[0]:end[0], start[1]:end[1], start[2]:end[2]]
        
        cline_binary = np.zeros_like(cline)
        cline_binary[cline == target_class_id] = 1
        cline = cline_binary

        # 【修改 3】同样彻底移除了中心线的 np.flip() 逻辑

        cline = transforms.apply_image(cline)
        dataset_dict["cline"] = torch.as_tensor(np.ascontiguousarray(cline[None, ...]))
        
        seg = transforms.apply_image(seg) 
        dataset_dict["seg"] = torch.as_tensor(np.ascontiguousarray(seg[None, ...]))

        return dataset_dict

class VesselSegDatasetMapper:
    """
    [恢复版] BBox 专用 Mapper
    逻辑：不翻转，不混合。严格根据 configs/bbox.yaml 里的 PRED_CLASS 来读取数据。
    """
    def __init__(self, cfg, transform_builder, is_train=True):
        self.is_train = is_train
        augmentations = transform_builder(cfg, is_train)
        self.augmentations = AugmentationList(augmentations)
        self.default_class_id = cfg.MODEL.PRED_CLASS

    def __call__(self, dataset_dict):
        dataset_dict = copy.deepcopy(dataset_dict)
        file_name = dataset_dict["file_name"]
        npz_file = np.load(file_name, allow_pickle=True)

        image = npz_file["img"].astype(np.float32)
        
        # 维度转置 (X,Y,Z) -> (Z,Y,X)
        image = image.transpose(2, 1, 0)
        
        seg = npz_file["seg"]
        seg = seg.transpose(2, 1, 0)
        
        # 只提取当前指定的类别
        target_class_id = self.default_class_id
        
        # 创建二值 Mask
        seg_binary = np.zeros_like(seg)
        seg_binary[seg == target_class_id] = 1
        seg = seg_binary

        # 下采样
        image = interpolate(image, type='img')
        seg = interpolate(seg, type='seg')

        # 同样这里也没有翻转逻辑

        aug_input = AugInput(image=image, sem_seg=seg)
        transforms = self.augmentations(aug_input)
        image = aug_input.image
        
        # 处理 Image
        image = torch.as_tensor(np.ascontiguousarray(image[None, ...]))
        dataset_dict["image"] = image

        # 处理 Seg (Mask)
        seg = transforms.apply_image(seg)
        dataset_dict["seg"] = torch.as_tensor(np.ascontiguousarray(seg[None, ...]))

        return dataset_dict