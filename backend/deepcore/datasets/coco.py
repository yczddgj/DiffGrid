from torchvision import datasets, transforms
from torch import tensor, long
import os

def COCO(data_path):
    channel = 3
    im_size = (None, None)  # COCO图像尺寸不固定
    num_classes = 80
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
    # 构建路径
    train_img_dir = os.path.join(data_path, 'train2017')
    train_ann_file = os.path.join(
        data_path, 'annotations', 'instances_train2017.json')
    val_img_dir = os.path.join(data_path, 'val2017')
    val_ann_file = os.path.join(
        data_path, 'annotations', 'instances_val2017.json')

    # 加载数据集
    dst_train = datasets.CocoDetection(root=train_img_dir,
                              annFile=train_ann_file,
                              transform=transform)

    dst_test = datasets.CocoDetection(root=val_img_dir,
                             annFile=val_ann_file,
                             transform=transform)
    # COCO类别名称
    class_names = [
        'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
        'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
        'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
        'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
        'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
        'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
        'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
        'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
        'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
        'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
    ]
    # coco不提供targets，需要自定义处理
    #dst_train.targets = tensor(dst_train.targets, dtype=long)
    #dst_test.targets = tensor(dst_test.targets, dtype=long)
    return channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test
