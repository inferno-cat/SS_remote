# 使用 torchvision 自带下载器
from torchvision.datasets import VOCSegmentation

VOCSegmentation(root='VOCdata', year='2012', image_set='train', download=True)
