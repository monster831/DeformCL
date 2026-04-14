from detectron2.data import DatasetCatalog, MetadataCatalog
from fvcore.common.file_io import PathManager
import os
import logging
logger = logging.getLogger(__name__)

def load_npz_cta_dataset(data_dir, size=None, filter_str=None):
    """
    加载数据集，支持通过 filter_str 过滤文件名
    """
    dataset_dicts = []
    files = PathManager.ls(data_dir)
    suffix = ".npz"
    # 1. 基础过滤：只保留 .npz
    files = [f for f in files if f.endswith(suffix)]
    
    # 2. 进阶过滤：只保留包含特定字符的文件 (如 "_L_")
    if filter_str is not None:
        files = [f for f in files if filter_str in f]
        
    files = sorted(files, reverse=False)
    if size is not None:
        files = files[:size]
    for f in files:
        basename = f[: -len(suffix)]
        ret = dict(
            file_name=os.path.join(data_dir, f),
            file_id=basename
        )
        dataset_dicts.append(ret)

    logger.info(f"Get {len(dataset_dicts)} data from {data_dir} (filter={filter_str}).")

    return dataset_dicts

# ========================================================
# 注册数据集
# ========================================================

# 🌟【新增】主动脉 (Aorta) 专属训练集
DatasetCatalog.register(
    'TotalSeg_Aorta_Train',
    lambda data_dir='./datasets/TotalSeg_Aorta/train':
    load_npz_cta_dataset(data_dir)
)
MetadataCatalog.get('TotalSeg_Aorta_Train').set(data_dir='./datasets/TotalSeg_Aorta/train')


# ========================================================
# 以下为您之前保留的髂动脉/股动脉注册信息，互不干扰
# ========================================================

DatasetCatalog.register(
    'TotalSeg_Iliac_Train',
    lambda data_dir='./datasets/TotalSeg_Iliac/train':
    load_npz_cta_dataset(data_dir)
)
MetadataCatalog.get('TotalSeg_Iliac_Train').set(data_dir='./datasets/TotalSeg_Iliac/train')

DatasetCatalog.register(
    'TotalSeg_Iliac_Train_L',
    lambda data_dir='./datasets/TotalSeg_Iliac/train':
    load_npz_cta_dataset(data_dir, filter_str='_L_')
)
MetadataCatalog.get('TotalSeg_Iliac_Train_L').set(data_dir='./datasets/TotalSeg_Iliac/train')

DatasetCatalog.register(
    'TotalSeg_Iliac_Train_R',
    lambda data_dir='./datasets/TotalSeg_Iliac/train':
    load_npz_cta_dataset(data_dir, filter_str='_R_')
)
MetadataCatalog.get('TotalSeg_Iliac_Train_R').set(data_dir='./datasets/TotalSeg_Iliac/train')

DatasetCatalog.register(
    'TotalSeg_Portal_Train',
    lambda data_dir='./datasets/TotalSeg_Portal/train':
    load_npz_cta_dataset(data_dir)
)
MetadataCatalog.get('TotalSeg_Portal_Train').set(data_dir='./datasets/TotalSeg_Portal/train')