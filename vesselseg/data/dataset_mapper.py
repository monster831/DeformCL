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
        tfm_gens.append(RandomCrop("absolute", crop_size))
        tfm_gens.append(RandomCrop("relative_range", (0.9, 0.9, 0.9)))
        tfm_gens.append(RandomFlip_Z(prob=0.5))
        # 现在 RandomFlip_X 已经在上面定义了，绝对不会报错
        tfm_gens.append(RandomFlip_X(prob=0.5))
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
        # 动态 ID 设置
        self.default_class_id = cfg.MODEL.PRED_CLASS

    def __call__(self, dataset_dict):
        dataset_dict = copy.deepcopy(dataset_dict)
        file_name = dataset_dict["file_name"]
        npz_file = np.load(file_name, allow_pickle=True)

        # ------------------------------------------------
        # 左右侧动态判断
        # ------------------------------------------------
        is_left_vessel = "_L_" in file_name or "Carotid_L" in file_name or "Left" in file_name
        target_class_id = 1 if is_left_vessel else 2

        pad = np.array([50, 50, 50])
        
        # 【关键修改 1】读取 cline 并转置维度 (X,Y,Z) -> (Z,Y,X)
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
        # 【关键修改 2】读取 image 并转置
        image = npz_file["img"].astype(np.float32).transpose(2, 1, 0)
        image = image[start[0]:end[0], start[1]:end[1], start[2]:end[2]]
        
        # 【关键修改 3】确认移除除法，防止双重归一化
        # image = image / 1024.

        # 【关键修改 4】读取 seg 并转置
        seg = npz_file["seg"].transpose(2, 1, 0)
        seg = seg[start[0]:end[0], start[1]:end[1], start[2]:end[2]]

        # 标签统一化 (都变 1)
        seg_binary = np.zeros_like(seg)
        seg_binary[seg == target_class_id] = 1 
        seg = seg_binary 

        # 空间归一化 (左变右)
        if is_left_vessel:
            image = np.flip(image, axis=2).copy()
            seg = np.flip(seg, axis=2).copy()

        aug_input = AugInput(image=image, sem_seg=seg)
        transforms = self.augmentations(aug_input)
        image = aug_input.image
        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image[None, ...]))

        cline = cline[start[0]:end[0], start[1]:end[1], start[2]:end[2]]
        
        cline_binary = np.zeros_like(cline)
        cline_binary[cline == target_class_id] = 1
        cline = cline_binary

        if is_left_vessel:
            cline = np.flip(cline, axis=2).copy()

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
        # 1. 恢复：直接使用配置里的 PRED_CLASS (1=左, 2=右)
        self.default_class_id = cfg.MODEL.PRED_CLASS

    def __call__(self, dataset_dict):
        dataset_dict = copy.deepcopy(dataset_dict)
        file_name = dataset_dict["file_name"]
        npz_file = np.load(file_name, allow_pickle=True)

        image = npz_file["img"].astype(np.float32)
        
        # 【关键修复 A】: 维度转置 (X,Y,Z) -> (Z,Y,X)
        # 让深度(Depth)变成第一维，符合模型预期
        image = image.transpose(2, 1, 0)
        #image = image / 1024.
        
        seg = npz_file["seg"]
        # 【关键修复 A】: 标签也要同步转置
        seg = seg.transpose(2, 1, 0)
        
        # =================================================
        # 【修改 1】只提取当前指定的类别，不做动态判断
        # 如果你运行命令行指定 PRED_CLASS 1，这里就只找 Label 1
        # =================================================
        target_class_id = self.default_class_id
        
        # 创建二值 Mask
        seg_binary = np.zeros_like(seg)
        seg_binary[seg == target_class_id] = 1
        seg = seg_binary

        # 下采样 (保持不变)
        image = interpolate(image, type='img')
        seg = interpolate(seg, type='seg')

        # =================================================
        # 【修改 2】删除了翻转逻辑 (Flip)
        # 现在不再把左侧翻转成右侧，保持原样
        # =================================================

        aug_input = AugInput(image=image, sem_seg=seg)
        transforms = self.augmentations(aug_input)
        image = aug_input.image
        
        # 处理 Image
        image = torch.as_tensor(np.ascontiguousarray(image[None, ...]))
        dataset_dict["image"] = image

        # 处理 Seg (Mask)
        seg = transforms.apply_image(seg)
        dataset_dict["seg"] = torch.as_tensor(np.ascontiguousarray(seg[None, ...]))

        # =================================================
        # ⚠️ 关键补充：生成 BBox 实例
        # 训练检测模型必须要有 instances，否则会报错。
        # =================================================
        #if seg is not None:
            # 获取图像尺寸 (D, H, W)
            #image_shape = image.shape[-3:] 
            # 生成检测需要的实例 (bbox, label)
            # 这里的 target_class_id 会作为 label 传进去
            #dataset_dict["instances"] = get_detection_dataset_dicts(
                #seg, np.where(seg > 0), target_class_id, image_shape
            #)

        return dataset_dict