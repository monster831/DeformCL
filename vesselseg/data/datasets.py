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
    
    # 2. 【核心修复】进阶过滤：只保留包含特定字符的文件 (如 "_L_")
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

# 1. 保留旧的 (Train)，但一般不建议用了
DatasetCatalog.register(
    'TotalSeg_Iliac_Train',
    lambda data_dir='./datasets/TotalSeg_Iliac/train':
    load_npz_cta_dataset(data_dir)
)
MetadataCatalog.get('TotalSeg_Iliac_Train').set(data_dir='./datasets/TotalSeg_Iliac/train')

# 2. 【新增】左侧专用训练集 (只读 _L_ 文件)
DatasetCatalog.register(
    'TotalSeg_Iliac_Train_L',
    lambda data_dir='./datasets/TotalSeg_Iliac/train':
    load_npz_cta_dataset(data_dir, filter_str='_L_')
)
MetadataCatalog.get('TotalSeg_Iliac_Train_L').set(data_dir='./datasets/TotalSeg_Iliac/train')

# 3. 【新增】右侧专用训练集 (只读 _R_ 文件)
DatasetCatalog.register(
    'TotalSeg_Iliac_Train_R',
    lambda data_dir='./datasets/TotalSeg_Iliac/train':
    load_npz_cta_dataset(data_dir, filter_str='_R_')
)
MetadataCatalog.get('TotalSeg_Iliac_Train_R').set(data_dir='./datasets/TotalSeg_Iliac/train')

# 4. 如果您有 test 文件夹，也可以照此办理
if os.path.exists('./datasets/TotalSeg_Iliac/test'):
    DatasetCatalog.register(
        'TotalSeg_Iliac_Test_L',
        lambda data_dir='./datasets/TotalSeg_Iliac/test':
        load_npz_cta_dataset(data_dir, filter_str='_L_')
    )
    MetadataCatalog.get('TotalSeg_Iliac_Test_L').set(data_dir='./datasets/TotalSeg_Iliac/test')