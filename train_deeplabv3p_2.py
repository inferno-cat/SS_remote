import os
import random
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
from torchvision.transforms import RandomCrop
from datetime import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt

# ====================== 优化后的 DeepLabV3Plus Model ======================

# Mish 激活函数
class Mish(nn.Module):
    def forward(self, x):
        return x * torch.tanh(F.softplus(x))

# SE 注意力模块
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

# 深度可分离卷积
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride,
                                   padding=padding, dilation=dilation, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.mish = Mish()

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return self.mish(x)

# 残差块
class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlock, self).__init__()
        self.conv1 = ConvBNReLU(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.conv2 = ConvBNReLU(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.se = SEBlock(out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.se(out)
        out += self.shortcut(x)
        return F.mish(out)

class ConvBNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.mish = Mish()

    def forward(self, x):
        return self.mish(self.bn(self.conv(x)))

class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ASPP, self).__init__()
        self.atrous_block1 = ConvBNReLU(in_channels, out_channels, kernel_size=1, padding=0, dilation=1)
        self.atrous_block4 = ConvBNReLU(in_channels, out_channels, kernel_size=3, padding=4, dilation=4)
        self.atrous_block8 = ConvBNReLU(in_channels, out_channels, kernel_size=3, padding=8, dilation=8)
        self.atrous_block12 = ConvBNReLU(in_channels, out_channels, kernel_size=3, padding=12, dilation=12)
        self.atrous_block16 = ConvBNReLU(in_channels, out_channels, kernel_size=3, padding=16, dilation=16)

        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            ConvBNReLU(in_channels, out_channels, kernel_size=1, padding=0)
        )

        self.conv1 = nn.Conv2d(out_channels * 6, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.mish = Mish()
        self.dropout = nn.Dropout(0.3)  # 降低 Dropout 率

    def forward(self, x):
        size = x.shape[2:]

        x1 = self.atrous_block1(x)
        x2 = self.atrous_block4(x)
        x3 = self.atrous_block8(x)
        x4 = self.atrous_block12(x)
        x5 = self.atrous_block16(x)

        x6 = self.global_avg_pool(x)
        x6 = F.interpolate(x6, size=size, mode='bilinear', align_corners=True)

        x = torch.cat((x1, x2, x3, x4, x5, x6), dim=1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.mish(x)
        return self.dropout(x)

class DeepLabV3Plus(nn.Module):
    def __init__(self, in_channels=3, num_classes=3):
        super(DeepLabV3Plus, self).__init__()

        # 增强的骨干网络
        self.layer1 = nn.Sequential(
            ResBlock(in_channels, 64, stride=1),
            ResBlock(64, 64, stride=1),
            nn.MaxPool2d(2, 2)  # Down to 1/2
        )
        self.layer2 = nn.Sequential(
            ResBlock(64, 128, stride=1),
            ResBlock(128, 128, stride=1),
            nn.MaxPool2d(2, 2)  # Down to 1/4
        )
        self.layer3 = nn.Sequential(
            ResBlock(128, 256, stride=1),
            ResBlock(256, 256, stride=1),
            nn.MaxPool2d(2, 2)  # Down to 1/8
        )
        self.layer4 = nn.Sequential(
            ResBlock(256, 512, stride=1),
            ResBlock(512, 512, stride=1),
            nn.MaxPool2d(2, 2)  # Down to 1/16
        )

        self.aspp = ASPP(512, 256)

        # 多级低级特征融合
        self.low_level_conv1 = ConvBNReLU(64, 48, kernel_size=1, padding=0)  # layer1 特征
        self.low_level_conv2 = ConvBNReLU(128, 48, kernel_size=1, padding=0)  # layer2 特征
        self.concat_conv1 = DepthwiseSeparableConv(256 + 48 + 48, 256)
        self.concat_conv2 = DepthwiseSeparableConv(256, 256)

        self.classifier = nn.Conv2d(256, num_classes, kernel_size=1)

    def forward(self, x):
        size = x.shape[2:]

        x1 = self.layer1(x)  # 1/2
        x2 = self.layer2(x1)  # 1/4
        x3 = self.layer3(x2)  # 1/8
        x4 = self.layer4(x3)  # 1/16

        x_aspp = self.aspp(x4)  # 1/16
        x_aspp = F.interpolate(x_aspp, size=x2.shape[2:], mode='bilinear', align_corners=True)

        x_low1 = self.low_level_conv1(F.interpolate(x1, size=x2.shape[2:], mode='bilinear', align_corners=True))
        x_low2 = self.low_level_conv2(x2)

        x = torch.cat((x_aspp, x_low1, x_low2), dim=1)
        x = self.concat_conv1(x)
        x = self.concat_conv2(x)

        x = self.classifier(x)
        x = F.interpolate(x, size=size, mode='bilinear', align_corners=True)
        return x

# ====================== 数据集定义 ======================

class SegmentationDataset(Dataset):
    def __init__(self, txt_file, base_dir, crop_size=(320, 320), is_train=True, resize_mode='crop'):
        self.is_train = is_train
        self.crop_w, self.crop_h = crop_size
        self.base_dir = base_dir
        self.resize_mode = resize_mode  # 'crop', 'resize', 'none'

        self.samples = []
        with open(txt_file, 'r', encoding='utf-8') as f:
            for line in f:
                img_path, mask_path = line.strip().split()
                self.samples.append((
                    os.path.join(base_dir, img_path),
                    os.path.join(base_dir, mask_path)
                ))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, mask_path = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path)

        if self.is_train:
            # 随机缩放（多尺度训练）
            if random.random() > 0.5:
                scale = random.uniform(0.75, 1.25)
                new_size = (int(640 * scale), int(640 * scale))
                image = TF.resize(image, size=new_size, interpolation=Image.BILINEAR)
                mask = TF.resize(mask, size=new_size, interpolation=Image.NEAREST)

            if self.resize_mode == 'crop':
                image = TF.resize(image, size=(max(self.crop_h, image.height), max(self.crop_w, image.width)))
                mask = TF.resize(mask, size=(max(self.crop_h, mask.height), max(self.crop_w, mask.width)), interpolation=Image.NEAREST)

                image = TF.to_tensor(image)
                mask = torch.from_numpy(np.array(mask)).long()

                i, j, h, w = RandomCrop.get_params(image, output_size=(self.crop_h, self.crop_w))
                image = TF.crop(image, i, j, h, w)
                mask = TF.crop(mask, i, j, h, w)

                if random.random() > 0.5:
                    image = TF.hflip(image)
                    mask = TF.hflip(mask)

            elif self.resize_mode == 'resize':
                image = TF.resize(image, size=(640, 640))
                mask = TF.resize(mask, size=(640, 640), interpolation=Image.NEAREST)

                image = TF.to_tensor(image)
                mask = torch.from_numpy(np.array(mask)).long()

                if random.random() > 0.5:
                    image = TF.hflip(image)
                    mask = TF.hflip(mask)

            elif self.resize_mode == 'none':
                image = TF.to_tensor(image)
                mask = torch.from_numpy(np.array(mask)).long()

            else:
                raise ValueError(f"Unsupported resize_mode: {self.resize_mode}")
        else:
            image = TF.to_tensor(image)
            mask = torch.from_numpy(np.array(mask)).long()

        return image, mask

# ====================== 工具函数 ======================

def get_num_classes(txt_path):
    with open(txt_path, 'r', encoding='utf-8') as f:
        return max(int(line.strip().split(':')[1]) for line in f) + 1

def compute_mIoU(pred, target, num_classes):
    pred = pred.detach().cpu().numpy()
    target = target.detach().cpu().numpy()
    ious = []
    for cls in range(num_classes):
        pred_inds = pred == cls
        target_inds = target == cls
        intersection = (pred_inds & target_inds).sum()
        union = (pred_inds | target_inds).sum()
        if union == 0:
            ious.append(float('nan'))
        else:
            ious.append(intersection / union)
    return np.nanmean(ious)

def save_prediction(image_tensor, pred_mask, save_path):
    image_np = image_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
    mask_np = pred_mask.cpu().numpy()

    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(image_np)
    plt.title("Image")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(mask_np, cmap='jet', vmin=0, vmax=mask_np.max())
    plt.title("Prediction")
    plt.axis('off')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, preds, targets):
        preds = F.softmax(preds, dim=1)
        targets_one_hot = F.one_hot(targets, num_classes=preds.size(1)).permute(0, 3, 1, 2).float()

        intersection = (preds * targets_one_hot).sum(dim=(2, 3))
        union = preds.sum(dim=(2, 3)) + targets_one_hot.sum(dim=(2, 3))
        dice_score = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice_score.mean()

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weight=None):
        super().__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, input, target):
        ce_loss = F.cross_entropy(input, target, weight=self.weight, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()

class HFLoss(nn.Module):
    def __init__(self, gamma=2.0, beta=0.7, delta=0.75):
        super().__init__()
        self.gamma = gamma
        self.beta = beta
        self.delta = delta
        self.epsilon = 1e-7

    def forward(self, preds, labels):
        preds = F.softmax(preds, dim=1)
        labels = F.one_hot(labels, num_classes=preds.shape[1]).permute(0, 3, 1, 2).float()

        loss = 0.0
        n = preds.shape[0]
        for i in range(n):
            p = preds[i]
            l = labels[i]
            tp = torch.sum(p * l)
            fp = torch.sum((p * (1 - l)) ** self.gamma)
            fn = torch.sum(((1 - p) * l) ** self.gamma)
            tversky = (tp + (1 - self.beta) * fp + self.beta * fn + self.epsilon) / (tp + self.epsilon)
            temp_loss = torch.pow(tversky, self.delta)
            temp_loss = torch.clamp(temp_loss, max=50.0)
            loss += temp_loss

        return loss / n

# ====================== 主函数 ======================

if __name__ == '__main__':
    base_dir = './data/Dataset1'
    train_txt = os.path.join(base_dir, 'train.txt')
    val_txt = os.path.join(base_dir, 'val.txt')
    class_map_txt = os.path.join(base_dir, 'class_mapping.txt')

    resize_mode = 'resize'  # 可选: 'crop', 'resize', 'none'
    num_classes = get_num_classes(class_map_txt)
    print(f"识别类别数：{num_classes}")
    print(f"图像尺寸处理方式：{resize_mode}")

    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    save_dir = f"DeepLabV3Plus-Optimized-{timestamp}"
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'val_preds'), exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_dataset = SegmentationDataset(train_txt, base_dir, is_train=True, resize_mode=resize_mode)
    val_dataset = SegmentationDataset(val_txt, base_dir, is_train=False)

    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=4, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=2)

    model = DeepLabV3Plus(in_channels=3, num_classes=num_classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)  # 使用 AdamW 优化器

    loss_type = "hfl"  # "ce", "focal", "dice", "hfl"

    if loss_type == "ce":
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)  # 添加标签平滑
    elif loss_type == "focal":
        criterion = FocalLoss()
    elif loss_type == "dice":
        criterion = DiceLoss()
    elif loss_type == "hfl":
        criterion = HFLoss()
    else:
        raise ValueError(f"不支持的损失类型: {loss_type}")

    num_epochs = 200
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        print(f"\n[Epoch {epoch+1}/{num_epochs}]")
        pbar = tqdm(train_loader, desc="Training")
        for imgs, masks in pbar:
            imgs, masks = imgs.to(device), masks.to(device)
            preds = model(imgs)
            loss = criterion(preds, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            pbar.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(train_loader)
        print(f"训练损失: {avg_loss:.4f}")

        model.eval()
        miou_scores = []
        with torch.no_grad():
            for idx, (img, mask) in enumerate(tqdm(val_loader, desc="Validation")):
                img = img.to(device)
                mask = mask.to(device)

                _, _, h, w = img.shape
                pad_h = (32 - h % 32) % 32
                pad_w = (32 - w % 32) % 32

                img_padded = F.pad(img, (0, pad_w, 0, pad_h), mode='reflect')
                pred = model(img_padded)
                pred = pred[:, :, :h, :w]
                pred_label = torch.argmax(pred, dim=1).squeeze(0)

                miou = compute_mIoU(pred_label, mask.squeeze(0), num_classes)
                miou_scores.append(miou)

                save_path = os.path.join(save_dir, 'val_preds', f'epoch{epoch+1}_img{idx+1}.png')
                save_prediction(img.cpu(), pred_label.cpu(), save_path)

        mean_miou = np.nanmean(miou_scores)
        print(f"验证 mIoU: {mean_miou:.4f}")