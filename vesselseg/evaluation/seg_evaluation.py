import logging
from collections import OrderedDict
from itertools import chain
import numpy as np
import torch
from detectron2.data.catalog import MetadataCatalog
from detectron2.evaluation import DatasetEvaluator
from detectron2.utils import comm
import os
from skimage import measure, morphology
import SimpleITK as sitk
from scipy.ndimage import distance_transform_edt

# =========================================================
#  辅助函数区域
# =========================================================

def get_dice_coeff_numpy(pred, target):
    """Numpy 版本的 Dice 计算"""
    smooth = 1.
    intersection = (pred * target).sum()
    return (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)

def compute_cldice(pred, gt):
    """计算骨架 Dice (clDice)"""
    if pred.sum() == 0 and gt.sum() == 0: return 1.0
    if pred.sum() == 0 or gt.sum() == 0: return 0.0
    skel_pred = morphology.skeletonize(pred)
    skel_gt = morphology.skeletonize(gt)
    tprec = (skel_pred * gt).sum() / (skel_pred.sum() + 1e-5)
    tsens = (skel_gt * pred).sum() / (skel_gt.sum() + 1e-5)
    return float(2 * tprec * tsens / (tprec + tsens + 1e-5))

def compute_hd95(pred, gt):
    """计算 95% 豪斯多夫距离 (HD95) - Scipy 稳定版"""
    if pred.sum() == 0 or gt.sum() == 0: return None 
    try:
        dt_gt = distance_transform_edt(1 - gt)
        dist_pred_to_gt = dt_gt[pred > 0]
        dt_pred = distance_transform_edt(1 - pred)
        dist_gt_to_pred = dt_pred[gt > 0]
        all_distances = np.concatenate([dist_pred_to_gt, dist_gt_to_pred])
        return np.percentile(all_distances, 95)
    except Exception as e:
        print(f"[HD95 Error]: {e}")
        return None

def compute_topology_errors(pred, gt):
    """计算拓扑误差 (Beta Error)"""
    euler_pred = measure.euler_number(pred, connectivity=3)
    euler_gt = measure.euler_number(gt, connectivity=3)
    euler_error = abs(euler_pred - euler_gt)
    
    labels_pred, num_pred = measure.label(pred, return_num=True, connectivity=3)
    labels_gt, num_gt = measure.label(gt, return_num=True, connectivity=3)
    beta0_error = abs(num_pred - num_gt)
    
    beta1_pred = num_pred - euler_pred
    beta1_gt = num_gt - euler_gt
    beta1_error = abs(beta1_pred - beta1_gt)
    
    return {'beta0_error': float(beta0_error), 'beta1_error': float(beta1_error), 'euler_error': float(euler_error)}


# =========================================================
#  评估器类定义
# =========================================================

class SegmentationBaseEvaluator(DatasetEvaluator):
    def __init__(self, dataset_name, cfg):
        self._dataset_name = dataset_name
        self._meta = MetadataCatalog.get(dataset_name)
        self._cpu_device = torch.device("cpu")
        self._logger = logging.getLogger(__name__)
        self._predictions = list()
        self.predictions = None

    def reset(self):
        self._predictions = list()

    def process(self, inputs, outputs):
        raise NotImplementedError

    def gather_predictions(self):
        all_predictions = comm.gather(self._predictions, dst=0)
        if not comm.is_main_process():
            return
        predictions = dict(chain.from_iterable(all_predictions))
        del all_predictions
        self.predictions = predictions

    def evaluate(self):
        raise NotImplementedError


class CommonDiceEvaluator(SegmentationBaseEvaluator):
    def __init__(self, dataset_name, cfg):
        super().__init__(dataset_name, cfg)
        self.model = cfg.MODEL.META_ARCHITECTURE
        self.pred_class = cfg.MODEL.PRED_CLASS
        
        if self.model == "Bbox3d":
            self.save_dir = f'bbox_pred{self.pred_class}'
            os.makedirs(self.save_dir, exist_ok=True)
        
    # 修改 vesselseg/evaluation/seg_evaluation.py 中的 process 方法

    def process(self, inputs, outputs):
        for inp, output in zip(inputs, outputs):
            file_id = inp['file_id']
            metrics = dict()
            
            # 1. 获取数据并转到 CPU Numpy
            seg_pred_raw = output['seg'].squeeze(0)
            seg_gt = inp['seg'].to(seg_pred_raw.device).squeeze(0)
            
            # 2. 二值化
            pred_mask = (seg_pred_raw > 0.5).float()
            gt_mask = seg_gt.gt(0).float()
            
            pred_np = pred_mask.cpu().numpy().astype(np.uint8)
            gt_np = gt_mask.cpu().numpy().astype(np.uint8)
            # ======================================================
            # 【修复版：最大连通域提取 (Post-processing)】
            # 过滤掉孤立的假阳性噪点，能大幅降低 HD95 和 beta0_error
            # ======================================================
            if pred_np.sum() > 0:
                from skimage import measure
                # 【关键修复】：必须显式指定 background=0，让它忽略背景！
                labels = measure.label(pred_np, connectivity=3, background=0)
                
                # 安全检查：确保图像中确实存在前景标签 (大于0)
                if labels.max() > 0:
                    # 此时 bincount 的索引 0 是背景，[1:] 全是真正的前景连通域
                    largest_cc_label = np.argmax(np.bincount(labels.flat)[1:]) + 1
                    pred_np = (labels == largest_cc_label).astype(np.uint8)
            # ======================================================
            
            # 3. 计算 Dice
            if gt_mask.sum() > 0:
                dice = get_dice_coeff_numpy(pred_np, gt_np)
            else:
                dice = 0.0 if pred_np.sum() > 0 else 1.0
            metrics['dice'] = dice

            # =========================================================
            # 【修复点 1】：如果是 Bbox3d 模型，计算 BBox 坐标
            # =========================================================
            full_metrics = {}
            
            if self.model == "Bbox3d":
                # 计算边界框: [min_x, max_x, min_y, max_y, min_z, max_z]
                # 注意：这里的维度顺序 (D, H, W) 需根据实际数据调整，通常 np.where 返回的是 (z, y, x) 或 (d, h, w)
                if pred_np.sum() > 0:
                    pos = np.where(pred_np > 0)
                    # 格式: [min_d, max_d, min_h, max_h, min_w, max_w]
                    bbox = [np.min(pos[0]), np.max(pos[0]), 
                            np.min(pos[1]), np.max(pos[1]), 
                            np.min(pos[2]), np.max(pos[2])]
                else:
                    # 如果没预测出东西，给一个默认全图或空值，防止报错
                    # 这里给全 0 可能会导致后续 crop 报错，建议给 shape 范围或者做容错
                    bbox = [0, pred_np.shape[0], 0, pred_np.shape[1], 0, pred_np.shape[2]]
                
                # 构造符合 dataset_mapper 读取的字典结构
                # mapper 读取路径: item[file_id]['bbox_pred']['bbox_pred']
                metrics['bbox_pred'] = bbox
                full_metrics['bbox_pred'] = metrics 
                # 同时保留 seg 键以兼容后续打印
                full_metrics['seg'] = dict(dice=dice)
            
            else:
                # 4. 计算其他高级指标 (仅非 BBox 模式)
                metrics['cldice'] = compute_cldice(pred_np, gt_np)
                metrics['hd95'] = compute_hd95(pred_np, gt_np)
                topo_metrics = compute_topology_errors(pred_np, gt_np)
                metrics.update(topo_metrics)
                full_metrics['seg'] = metrics

            self._predictions.append((file_id, full_metrics))
            

    def evaluate(self):
        if self.predictions is None:
            self.gather_predictions()
        if not comm.is_main_process():
            return
        
        ret = OrderedDict()
        metrics_list = list(self.predictions.values())
        
        if not metrics_list:
            print("No predictions found!")
            return ret

        # =========================================================
        # 【修复点 2】：保存预测结果到 .npz 文件
        # =========================================================
        if self.model == "Bbox3d":
            save_path = os.path.join(self.save_dir, 'bbox_pred.npz')
            # self.predictions 是一个 dict: {file_id: full_metrics, ...}
            # dataset_mapper 中读取方式是: item = np.load(...)['metrics'].item()
            np.savez(save_path, metrics=self.predictions)
            self._logger.info(f"BBox predictions saved to: {save_path}")

        # 以下是原有的打印逻辑，保持不变
        first_metric = metrics_list[0]['seg']
        keys = ['dice', 'cldice', 'hd95', 'beta0_error', 'beta1_error', 'euler_error']
        keys = [k for k in keys if k in first_metric]

        print("="*60)
        print(f"{'Metric':<15} | {'Mean':<10}")
        print("-" * 30)
        
        for k in keys:
            values = [m['seg'][k] for m in metrics_list]
            values = [v for v in values if v is not None]
            if len(values) > 0:
                mean_val = np.mean(values)
                print(f"{k:<15} | {mean_val:.4f}")
                ret[f"seg/{k}"] = mean_val
            else:
                print(f"{k:<15} | NaN")
        print("="*60)
            
        return ret