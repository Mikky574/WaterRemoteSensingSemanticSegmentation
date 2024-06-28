import torch
from torch.utils.data import DataLoader
from models import UNetPlusPlus, UNet
from prepocess import load_data_paths, dset_Dataset
from utils import PolyLRScheduler, dice_loss, calculate_dice, update_ema,load_cloud_images
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
import random
import numpy as np


def copy_paste_batch(images, masks, copy_rate=0.5, square_size=256):
    """
    Apply Copy-Paste augmentation to a batch of images and masks.

    Args:
    images (torch.Tensor): Tensor of shape (B, 6, 256, 256).
    masks (torch.Tensor): Tensor of shape (B, 1, 256, 256).
    copy_rate (float): Probability of applying the augmentation to a sample.
    square_size (int): Size of the square to be copied.

    Returns:
    torch.Tensor, torch.Tensor: Augmented images and masks.
    """
    # Clone images and masks to avoid modifying the original data
    augmented_images = images.clone()
    augmented_masks = masks.clone()

    B, C, H, W = images.shape

    augmentation = False

    if random.random() < copy_rate:
        for i in range(B):
            # Choose a random image as source
            src_idx = (i+1) % B

            # Randomly select top-left corner of the square
            top_x = random.randint(0, W - square_size)
            top_y = random.randint(0, H - square_size)

            # Copy the square from source to target
            augmented_images[i, :, top_y:top_y+square_size, top_x:top_x+square_size] = \
                images[src_idx, :, top_y:top_y +
                       square_size, top_x:top_x+square_size]

            augmented_masks[i, :, top_y:top_y+square_size, top_x:top_x+square_size] = \
                masks[src_idx, :, top_y:top_y +
                      square_size, top_x:top_x+square_size]

        augmentation = True

    return augmented_images, augmented_masks, augmentation


def cloud_augmentation(images, cloud_images, num_clouds=1, copy_rate=0.5):
    """
    随机选择整个云图像，并将其覆盖到训练图像上。云图像可能会随机翻转。

    Args:
    images (torch.Tensor): 训练图像的批次，形状为 (B, C, H, W)。
    cloud_images (list of np.ndarray): 云图像的列表，其中云以 -999 表示背景。
    num_clouds (int): 每个图像中要添加的云的数量。
    copy_rate (float): 应用增强的概率。

    Returns:
    torch.Tensor: 增强后的图像。
    """
    augmented_images = images.clone()
    B, C, H, W = images.shape

    for i in range(B):
        if random.random() < copy_rate:
            for _ in range(num_clouds):
                # 随机选择一个云图像
                cloud = random.choice(cloud_images)

                # 随机翻转云图像
                if random.random() < 0.5:  # 50% 的概率水平翻转
                    cloud = np.flip(cloud, axis=2)
                if random.random() < 0.5:  # 50% 的概率垂直翻转
                    cloud = np.flip(cloud, axis=1)

                cloud_h, cloud_w = cloud.shape[1:]

                # 随机选择目标图像的位置
                target_x = random.randint(0, W - cloud_w)
                target_y = random.randint(0, H - cloud_h)

                # 创建云图像的掩码，只复制云区域
                cloud_mask = (cloud != -999)

                # 确保 NumPy 数组具有正步长
                cloud = np.ascontiguousarray(cloud)
                cloud_mask = np.ascontiguousarray(cloud_mask)

                # 将云图像覆盖到目标图像上
                for c in range(C):
                    augmented_images[i, c, target_y:target_y+cloud_h, target_x:target_x+cloud_w] = \
                        torch.where(torch.from_numpy(cloud_mask[c]), torch.from_numpy(
                            cloud[c]), augmented_images[i, c, target_y:target_y+cloud_h, target_x:target_x+cloud_w])

    return augmented_images


class Trainer:
    def __init__(self, model, device, train_loader, val_loader, lr, weight_decay, num_epochs, results_dir, resume=False):
        self.model = model
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.lr = lr
        self.optimizer = torch.optim.SGD(model.parameters(), self.lr, weight_decay=weight_decay,
                                         momentum=0.99, nesterov=True)
        self.lr_scheduler = PolyLRScheduler(self.optimizer, self.lr, num_epochs)
        self.num_epochs = num_epochs
        self.results_dir = results_dir
        self.best_ema_dice = 0

        # 初始化结果存储列表
        self.train_losses = []
        self.val_losses = []
        self.train_dices = []
        self.val_dices = []
        self.ema_dices = []

        os.makedirs(results_dir, exist_ok=True)
        self.log_file = os.path.join(results_dir, 'log.txt')
        self.resume = resume
        self.current_epoch = 0
        if resume:
            self.load_previous_state()
        else:
            self.lr_scheduler = PolyLRScheduler(self.optimizer, self.lr, self.num_epochs)

    def load_previous_state(self):
        # 从log文件中读取历史训练数据
        with open(self.log_file, 'r') as file:
            lines = file.readlines()

        self.train_losses, self.val_losses, self.train_dices, self.val_dices, self.ema_dices = [], [], [], [], []
        for line in lines:
            if 'Train Loss' in line:
                parts = line.split(',')
                train_loss = float(parts[0].split(': ')[1])
                train_dice = float(parts[1].split(': ')[1])
                val_loss = float(parts[2].split(': ')[1])
                val_dice = float(parts[3].split(': ')[1])
                ema_dice = float(parts[4].split(': ')[1])

                self.train_losses.append(train_loss)
                self.val_losses.append(val_loss)
                self.train_dices.append(train_dice)
                self.val_dices.append(val_dice)
                self.ema_dices.append(ema_dice)

        # 更新最佳EMA Dice值
        if self.ema_dices:
            self.best_ema_dice = max(self.ema_dices)

        # 加载之前保存的模型权重
        last_model_path = os.path.join(self.results_dir, 'last.pth')
        if os.path.exists(last_model_path):
            self.model.load_state_dict(torch.load(last_model_path))
            print("Loaded model weights from 'last.pth'.")

        # 确定当前epoch
        self.current_epoch = len(self.train_losses)
        for _ in range(self.current_epoch):
            self.lr_scheduler.step()
            
        print(f"Updated learning rate for epoch {self.current_epoch + 1}.")

    def train_epoch(self):
        self.model.train()
        train_loss, train_dice = 0.0, 0.0
        # 云图像的根目录
        cloud_root = r'F:\需要认真总结个项目了\敲代码\data\npz\cloud'
        cloud_images = load_cloud_images(cloud_root)
        for images, labels in tqdm(self.train_loader, desc='Training'):
            images, labels, augmentation = copy_paste_batch(images, labels)
            if not augmentation:
                images = cloud_augmentation(
                    images, cloud_images, num_clouds=random.choice(range(3)), copy_rate=0.5)
            images, labels = images.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = dice_loss(outputs, labels)
            dice_score = calculate_dice(outputs, labels)
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()
            train_dice += dice_score.item()

        return train_loss / len(self.train_loader), train_dice / len(self.train_loader)

    def validate_epoch(self):
        self.model.eval()
        val_loss, val_dice = 0.0, 0.0
        with torch.no_grad():
            for images, labels in tqdm(self.val_loader, desc='Validation'):
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss = dice_loss(outputs, labels)
                dice_score = calculate_dice(outputs, labels)
                val_loss += loss.item()
                val_dice += dice_score.item()

        return val_loss / len(self.val_loader), val_dice / len(self.val_loader)

    def log_print(self, s):
        print(s)
        with open(self.log_file, 'a', encoding="utf-8") as log:
            print(s, file=log)

    def plot_results(self):
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.plot(self.train_losses, label='Train Loss')
        plt.plot(self.val_losses, label='Val Loss')
        plt.title('Loss over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(self.train_dices, label='Train Dice')
        plt.plot(self.val_dices, label='Val Dice')
        plt.plot(self.ema_dices, label='EMA Dice', linestyle='--')
        plt.title('Dice Score over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Dice Score')
        plt.legend()

        plt.savefig(os.path.join(self.results_dir, f'results_.png'))
        plt.close()

    def train(self):
        start_epoch = len(self.train_losses) if self.resume else 0
        for epoch in range(start_epoch, self.num_epochs):
            current_lr = self.optimizer.param_groups[0]['lr']
            self.log_print(f"Epoch {epoch+1}/{self.num_epochs}")
            self.log_print(f"Current LR: {current_lr:.8f}")

            train_loss, train_dice = self.train_epoch()
            val_loss, val_dice = self.validate_epoch()

            # 更新结果列表
            self.train_losses.append(train_loss)
            self.train_dices.append(train_dice)
            self.val_losses.append(val_loss)
            self.val_dices.append(val_dice)

            ema_dice = update_ema(
                self.ema_dices[-1], val_dice) if self.ema_dices else val_dice
            self.ema_dices.append(ema_dice)

            if ema_dice > self.best_ema_dice:
                self.best_ema_dice = ema_dice
                self.log_print(
                    f"🎉 New best EMA Dice achieved: {ema_dice:.4f} 🎉")
                torch.save(self.model.state_dict(), os.path.join(
                    self.results_dir, 'best.pth'))

            torch.save(self.model.state_dict(), os.path.join(
                self.results_dir, 'last.pth'))
            self.lr_scheduler.step()

            self.plot_results()
            self.log_print(
                f"Train Loss: {train_loss:.4f}, Train Dice: {train_dice:.4f}, Val Loss: {val_loss:.4f}, Val Dice: {val_dice:.4f}, EMA Dice: {ema_dice:.4f}")


if __name__ == '__main__':
    # Define hyperparameters
    initial_lr = 0.01
    weight_decay = 3e-5
    num_epochs = 1000
    batch_size = 16
    fold_number = 1
    results_dir = f'results_folder{fold_number}'
    resume = False  # 继续训练？
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root_path = r'F:\敲代码\data\npz\train_pieces'
    train_data_paths, val_data_paths = load_data_paths(root_path, fold_number)

    train_dataset = dset_Dataset(train_data_paths, transform=True)
    val_dataset = dset_Dataset(val_data_paths)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    model = UNetPlusPlus(model_name='mobilenet_v2').get_model().to(device)
    trainer = Trainer(model, device, train_loader, val_loader,
                      initial_lr, weight_decay, num_epochs, results_dir, resume)
    trainer.train()
