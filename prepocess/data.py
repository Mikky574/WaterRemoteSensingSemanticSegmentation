import torch
from torch.utils.data import Dataset
import numpy as np
import json
import os
import random
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF


def load_data_paths(root_path, fold_number):
    """
    从指定根目录加载数据路径。

    参数:
    root_path - 数据的根目录。
    fold_number - 选择的数据折数。

    返回:
    训练数据和验证数据的文件路径列表。
    """
    with open(os.path.join(root_path, 'folds.json'), 'r') as file:
        folds = json.load(file)

    # 获取指定折的训练和验证集
    fold_key = f'fold_{fold_number}'
    train_datasets = folds[fold_key]['train']
    val_datasets = folds[fold_key]['val']

    # 收集训练数据文件路径
    train_data_paths = []
    for dataset_name in train_datasets:
        dataset_path = os.path.join(root_path, dataset_name)
        for file_name in os.listdir(dataset_path):
            if file_name.endswith('.npz'):
                train_data_paths.append(os.path.join(dataset_path, file_name))

    # 收集验证数据文件路径
    val_data_paths = []
    for dataset_name in val_datasets:
        dataset_path = os.path.join(root_path, dataset_name)
        for file_name in os.listdir(dataset_path):
            if file_name.endswith('.npz'):
                val_data_paths.append(os.path.join(dataset_path, file_name))

    return train_data_paths, val_data_paths


def transform_both(image, label, transform_func, *args, **kwargs):
    """
    同时对图像和标签应用相同的变换。

    参数:
    image - 输入图像。
    label - 对应的标签。
    transform_func - 变换函数。
    *args, **kwargs - 变换函数的额外参数。

    返回:
    变换后的图像和标签。
    """
    image = transform_func(image, *args, **kwargs)
    label = transform_func(label, *args, **kwargs)
    return image, label


class dset_Dataset(Dataset):
    """
    自定义的数据集类，用于加载和变换数据。

    参数:
    data_paths - 数据文件路径列表。
    transform - 是否应用数据增强。
    """

    def __init__(self, data_paths, transform=None):
        self.data_paths = data_paths
        self.transform = transform

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, idx):
        file_path = self.data_paths[idx]
        data = np.load(file_path)
        image = torch.from_numpy(data['images']).float()
        label = torch.from_numpy(data['label']).float()

        if self.transform:
            image, label = self.apply_transforms(image, label)

        return image, label

    def apply_transforms(self, image, label, resized_crop_size=(256, 256)):
        # Random rotation
        if random.random() > 0.5:
            angle = random.choice([90, 180, 270])
            image, label = transform_both(
                image, label, TF.rotate, angle=angle)  # 旋转
        # Random horizontal flip
        if random.random() > 0.5:
            image, label = transform_both(image, label, TF.hflip)

        # Random vertical flip
        if random.random() > 0.5:
            image, label = transform_both(image, label, TF.vflip)

        # RandomResizedCrop
        i, j, h, w = transforms.RandomResizedCrop.get_params(
            image, scale=(0.8, 1.0), ratio=(1, 1))
        image, label = transform_both(image, label, TF.resized_crop, i, j, h, w,
                                      size=resized_crop_size,antialias=True)  # 假设您的原始大小是 256x256
        return image, label


if __name__ == "__main__":
    root_path = 'D:\\需要认真总结个项目了\\敲代码\\data\\npz\\train_pieces'
    fold_number = 1
    train_data_paths, val_data_paths = load_data_paths(root_path, fold_number)
    train_dataset = dset_Dataset(train_data_paths, transform=True)
    val_dataset = dset_Dataset(val_data_paths)
    print(f"{fold_number}折的训练集大小:", len(train_dataset))
    print(f"{fold_number}验证集大小:", len(val_dataset))
    print("第一个训练样本的标签和图像形状:", train_dataset[0]
          [1].shape, train_dataset[0][0].shape)
    
