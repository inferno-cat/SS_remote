import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
from torchvision.models.segmentation import deeplabv3_resnet50
from torchvision.datasets import VOCSegmentation
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. 数据预处理和加载
# 定义数据变换，确保图像和标签尺寸一致
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((520, 520)),  # 调整图像到固定尺寸
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((520, 520)),  # 调整图像到固定尺寸
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
}

# 自定义标签变换（与图像变换保持一致）
target_transforms = {
    'train': transforms.Compose([
        transforms.Resize((520, 520), interpolation=Image.NEAREST),  # 使用最近邻插值以保持整数标签
    ]),
    'val': transforms.Compose([
        transforms.Resize((520, 520), interpolation=Image.NEAREST),
    ])
}

# 自定义数据集类
class VOCSegmentationCustom(VOCSegmentation):
    def __init__(self, root, year, image_set, transform=None, target_transform=None):
        super().__init__(root, year=year, image_set=image_set, download=False)
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        img, target = super().__getitem__(index)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        target = torch.as_tensor(np.array(target), dtype=torch.long)
        target[target == 255] = 21  # 将忽略区域转换为背景类
        return img, target


# 定义训练和验证函数
def train_one_epoch(model, data_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for images, targets in tqdm(data_loader, desc="Training"):
        images, targets = images.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(images)['out']
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
    epoch_loss = running_loss / len(data_loader.dataset)
    return epoch_loss


def validate(model, data_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for images, targets in tqdm(data_loader, desc="Validating"):
            images, targets = images.to(device), targets.to(device)
            outputs = model(images)['out']
            loss = criterion(outputs, targets)
            running_loss += loss.item() * images.size(0)
    val_loss = running_loss / len(data_loader.dataset)
    return val_loss


def visualize_segmentation(model, data_loader, device, num_images=2):
    model.eval()
    images, targets = next(iter(data_loader))
    images, targets = images.to(device), targets.to(device)
    with torch.no_grad():
        outputs = model(images)['out']
        preds = torch.argmax(outputs, dim=1)
    for i in range(num_images):
        img = images[i].cpu().permute(1, 2, 0).numpy()
        pred = preds[i].cpu().numpy()
        target = targets[i].cpu().numpy()
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 3, 1)
        plt.title("Input Image")
        plt.imshow(img)
        plt.subplot(1, 3, 2)
        plt.title("Ground Truth")
        plt.imshow(target)
        plt.subplot(1, 3, 3)
        plt.title("Prediction")
        plt.imshow(pred)
        plt.show()


# 主程序
if __name__ == '__main__':
    # 加载数据集
    train_dataset = VOCSegmentationCustom(
        root='VOCdata', year='2012', image_set='train',
        transform=data_transforms['train'], target_transform=target_transforms['train']
    )
    val_dataset = VOCSegmentationCustom(
        root='VOCdata', year='2012', image_set='val',
        transform=data_transforms['val'], target_transform=target_transforms['val']
    )
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=4)

    # 定义模型
    model = deeplabv3_resnet50(weights='COCO_WITH_VOC_LABELS_V1')
    model.classifier[4] = nn.Conv2d(256, 21, kernel_size=(1, 1), stride=(1, 1))
    model = model.to(device)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss(ignore_index=21)
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0005)

    # 训练循环
    num_epochs = 10
    best_loss = float('inf')
    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        print(f'Train Loss: {train_loss:.4f}')
        val_loss = validate(model, val_loader, criterion, device)
        print(f'Validation Loss: {val_loss:.4f}')
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), 'deeplabv3_voc_best.pth')
            print("Saved best model")

    print("Training completed!")
    visualize_segmentation(model, val_loader, device)