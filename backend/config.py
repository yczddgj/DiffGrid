import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
_data_root = Path(os.environ.get('DIFFGRID_DATA_DIR', 'database')).expanduser()
if not _data_root.is_absolute():
    _data_root = PROJECT_ROOT / _data_root
DATABASE_DIR = _data_root.resolve()
INDEX_DIR = str(DATABASE_DIR / 'index')
DATASET_DIR = str(DATABASE_DIR / 'datasets')
SLICE_DIR = str(DATABASE_DIR / 'slices')
TRAINING_DIR = str(DATABASE_DIR / 'training_results')
REDUCED_DIR = str(DATABASE_DIR / 'reduced')
RESULT_DIR = str(DATABASE_DIR / 'results')
RAW_DIR = str(DATABASE_DIR / 'rawdata')
MODEL_DIR = Path(os.environ.get('DIFFGRID_MODEL_DIR', str(PROJECT_ROOT / 'backend' / 'models'))).expanduser()
if not MODEL_DIR.is_absolute():
    MODEL_DIR = PROJECT_ROOT / MODEL_DIR
BACKEND_HOST = os.environ.get('DIFFGRID_HOST', '127.0.0.1')
BACKEND_PORT = int(os.environ.get('DIFFGRID_PORT', '5000'))
USE_REGULARIZATION = os.environ.get('DIFFGRID_REGULARIZATION', '1') != '0'
GRID_RESOLUTION = int(os.environ.get('DIFFGRID_GRID_RESOLUTION', '400'))
MODE = 'default'
modes=[
    "scatter",
    "kde",
    "kde-scatter",
    "default",
    "lsdd"
]

# 降维参数
fit_mode = ['joint', 's1', 's2', 'coreset']
n_neighbors = int(os.environ.get('DIFFGRID_N_NEIGHBORS', '100'))
feature_type = os.environ.get('DIFFGRID_FEATURE_TYPE', 'dreamsim')
extraction_methods = [
    'resnet50',
    'clip',
    'dreamsim'
]

data_path = 'rawdata'
image_path = 'datasets'
result_path = 'results'
coreset_methods = [
    #    'Cal',
    'ContextualDiversity',
    'Craig',
    'DeepFool',
    # 'EarlyTrain',
    'Forgetting',
    'Full',
    'Glister',
    'GradMatch',
    'GraNd',
    'Herding',
    'kCenterGreedy',
    'Submodular',
    'Uncertainty',
    'Uniform']

training_models = [
    "AlexNet",       # 来自 alexnet.py
    "InceptionV3",   # 来自 inceptionv3.py
    "LeNet",         # 来自 lenet.py
    "MLP",           # 来自 mlp.py
    "MobileNetV3",   # 来自 mobilenetv3.py
    "ResNet18",      # 来自 resnet.py
    "ResNet34",      # 来自 resnet.py
    "ResNet50",      # 来自 resnet.py
    "ResNet101",     # 来自 resnet.py
    "ResNet152",     # 来自 resnet.py
    "VGG11",         # 来自 vgg.py
    "VGG13",         # 来自 vgg.py
    "VGG16",         # 来自 vgg.py
    "VGG19",         # 来自 vgg.py
    "WRN168",        # 来自 wideresnet.py
    "WRN2810",       # 来自 wideresnet.py
    "WRN282",        # 来自 wideresnet.py
    "WRN502",        # 来自 wideresnet.py
    "WRN1012"        # 来自 wideresnet.py
]


dataset_names_classification = [
    "CIFAR10",
    "CIFAR100",
    "FashionMNIST",
    # "ImageNet",
    "MNIST",
    "QMNIST",
    "SVHN",
    "TinyImageNet"
]

dataset_names_object_detection = [
    "EAGLE",
    "COCO",
]

seeds = [
    114,
    # 2025,
    # 3000,
    486,
    5656
]
