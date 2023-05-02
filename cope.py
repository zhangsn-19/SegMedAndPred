import os
import cv2
import json
import ipdb
import numpy as np
import os
import json
import torch
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image, ImageDraw
from torchvision import transforms
from torch.utils.data import Dataset

train_transforms = transforms.Compose([
    transforms.RandomRotation(degrees=30),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.RandomCrop(size=128),
    transforms.Resize(size=(256, 256)),
    transforms.ToTensor(),
])

class ScarDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_filenames = [x for x in os.listdir(root_dir) if x.endswith('.jpg')]
        self.augmentation_transform = A.Compose([
            A.RandomResizedCrop(256, 256),
            A.HorizontalFlip(),
            A.VerticalFlip(),
            A.Rotate(limit=30),
            A.RandomBrightnessContrast(),
            A.Normalize(),
            ToTensorV2()
        ])

    def __len__(self):
        return len(self.image_filenames * 5)

    def __getitem__(self, index):
        image_filename = self.image_filenames[index // 5]
        image_path = os.path.join(self.root_dir, image_filename)
        image = Image.open(image_path).convert('RGB')

        # Resize image and mask to same size
        resize_transform = transforms.Resize((256, 256))  # Specify desired size
        image = resize_transform(image)

        # if self.transform:
        #     image = self.transform(image)

        # Generate the mask filenames from the image filename
        mask_filename = image_filename.split('.')[0] + '.json'
        mask_path = os.path.join(self.root_dir, mask_filename)

        # Load the mask from json
        with open(mask_path, 'r') as f:
            mask_data = json.load(f)
        body_mask_points = mask_data['shapes'][0]['points']
        boundary_mask_points = mask_data['shapes'][1]['points']

        # Create binary masks from points
        # Resize mask to match image size
        image_size = image.size
        body_mask = self.create_binary_mask(image_size, body_mask_points)
        body_mask = resize_transform(body_mask)
        boundary_mask = self.create_binary_mask(image_size, boundary_mask_points)
        boundary_mask = resize_transform(boundary_mask)

        # Convert PIL.Image.Image to torch.Tensor and ensure data types are consistent
        image = transforms.ToTensor()(image)
        body_mask = transforms.ToTensor()(body_mask)  # Convert mask to tensor
        boundary_mask = transforms.ToTensor()(boundary_mask)  # Convert mask to tensor
        
        if self.transform:
            # Convert PyTorch tensor to NumPy array and specify data type as np.uint8
            image_np = (image.numpy() * 255).clip(0, 255).astype(np.uint8)
            body_mask_np = (body_mask.numpy() * 255).clip(0, 255).astype(np.uint8)
            boundary_mask_np = (boundary_mask.numpy() * 255).clip(0, 255).astype(np.uint8)

            image_np = np.transpose(image_np,(1,2,0))
            body_mask_np = np.transpose(body_mask_np,(1,2,0)).squeeze(-1)
            boundary_mask_np = np.transpose(boundary_mask_np,(1,2,0)).squeeze(-1)
            # Convert NumPy arrays to PIL Images
            # print(image_np.shape, body_mask_np.shape, boundary_mask_np.shape)
            # image_pil = Image.fromarray(image_np)
            # body_mask_pil = Image.fromarray(body_mask_np)
            # boundary_mask_pil = Image.fromarray(boundary_mask_np)

            # Now you can use the PIL Images in your code
            # For example, you can pass them to the augmentation_transform function
            packed = self.augmentation_transform(image=image_np, 
                                                body_mask=body_mask_np, 
                                                boundary_mask=boundary_mask_np)
            image = packed['image']
            body_mask = packed['body_mask']
            boundary_mask = packed['boundary_mask']

        return image, body_mask, boundary_mask

    def create_binary_mask(self, image_size, points):
        mask = Image.new('L', image_size, 0)
        ImageDraw.Draw(mask).polygon([tuple(p) for p in points], outline=1, fill=1)
        return mask


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

# 定义 UNet 网络模型
class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.middle = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, out_channels, kernel_size=2, stride=2)
        )

    def forward(self, x):
        x1 = self.encoder(x)
        x2 = self.middle(x1)
        x3 = self.decoder(x2)
        return x3

import torch.nn.functional as F

# 定义训练函数
def train(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0

    for i, (image, body_mask, boundary_mask) in enumerate(dataloader):

        image = image.to(device)
        body_mask = body_mask.to(device)
        boundary_mask = boundary_mask.to(device)
        body_mask = body_mask.unsqueeze(1)
        boundary_mask = boundary_mask.unsqueeze(1)
        body_mask = F.interpolate(body_mask, size=(128, 128), mode='bilinear', align_corners=False)
        boundary_mask = F.interpolate(boundary_mask, size=(128, 128), mode='bilinear', align_corners=False)
        optimizer.zero_grad()
        outputs = model(image)
        # ipdb.set_trace()
        loss = criterion(outputs, (body_mask / 255.0).to(torch.float32)) + criterion(outputs, (boundary_mask / 255.0).to(torch.float32))
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)

# 定义测试函数
def test(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for i, (image, body_mask, boundary_mask) in enumerate(dataloader):
            image = image.to(device)
            body_mask = body_mask.to(device)
            boundary_mask = boundary_mask.to(device)
            body_mask = body_mask.unsqueeze(1)
            boundary_mask = boundary_mask.unsqueeze(1)
            body_mask = F.interpolate(body_mask, size=(128, 128), mode='bilinear', align_corners=False)
            boundary_mask = F.interpolate(boundary_mask, size=(128, 128), mode='bilinear', align_corners=False)
            outputs = model(image)
            loss = criterion(outputs, (body_mask / 255.0).to(torch.float32)) + criterion(outputs, (boundary_mask / 255.0).to(torch.float32))
        
            total_loss += loss.item()

    return total_loss / len(dataloader)

if __name__ == "__main__":
    # 设置超参数
    batch_size = 4
    learning_rate = 0.001
    epochs = 10

    # 实例化数据集对象
    dataset = ScarDataset(root_dir='.', transform=train_transforms)

    # 划分训练集和测试集
    # train_dataset, test_dataset = train_test_split(dataset, test_size=0.2, random_state=42)

    # 实例化数据加载器
    train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # 实例化 UNet 网络
    in_channels = 3 # 输入图像通道数
    out_channels = 1 # 输出mask通道数
    model = UNet(in_channels, out_channels)

    # 定义损失函数和优化器
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 将模型移动到GPU（如果可用）
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 训练模型
    for epoch in range(epochs):
        train_loss = train(model, train_dataloader, criterion, optimizer, device)
        test_loss = test(model, test_dataloader, criterion, device)
        print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")

    # 使用训练好的模型进行图像分割
    model.eval()

    # 从测试集中随机选择一张图像
    random_index = torch.randint(0, len(dataset), (1,))
    image, body_mask, boundary_mask = dataset[random_index]

    # 将图像移动到GPU（如果可用）
    image = image.to(device)

    # 前向传播
    outputs = model(image)
    body_mask_pred = torch.sigmoid(outputs[:, 0, :, :]) # sigmoid激活函数将输出转换为概率
    boundary_mask_pred = torch.sigmoid(outputs[:, 1, :, :])

    # 将预测的mask转换为二值图像
    body_mask_pred = (body_mask_pred > 0.5).float()
    boundary_mask_pred = (boundary_mask_pred > 0.5).float()

    # 可视化原始图像、真实mask和预测mask
    import matplotlib.pyplot as plt
    plt.subplot(1, 3, 1)
    plt.imshow(image.cpu().permute(1, 2, 0))
    plt.title('Image')
    plt.subplot(1, 3, 2)
    plt.imshow(body_mask.cpu().squeeze(), cmap='gray')
    plt.title('Ground Truth BODY Mask')
    plt.subplot(1, 3, 3)
    plt.imshow(body_mask_pred.cpu().squeeze(), cmap='gray')
    plt.title('Predicted BODY Mask')
    plt.show()

    plt.subplot(1, 3, 1)
    plt.imshow(image.cpu().permute(1, 2, 0))
    plt.title('Image')
    plt.subplot(1, 3, 2)
    plt.imshow(boundary_mask.cpu().squeeze(), cmap='gray')
    plt.show()

