import torch
import numpy as np
import os
from PIL import Image
from models import UNetPlusPlus
import rasterio
from tqdm import tqdm

import rasterio
from rasterio.transform import from_origin

def save_inference_as_tiff(output, original_tif_path, target_path):
    with rasterio.open(original_tif_path) as src:
        transform = src.transform
        crs = src.crs
        meta = src.meta

    output_np = (output - output.min()) / (output.max() - output.min()) * 255
    output_np = output_np.astype(rasterio.uint8)

    meta.update({"driver": "GTiff",
                 "height": output_np.shape[0],
                 "width": output_np.shape[1],
                 "count": 1,
                 "dtype": 'uint8'})

    with rasterio.open(target_path, 'w', **meta) as dst:
        dst.write(output_np, 1)

def load_model(model_path, device):
    model = UNetPlusPlus(model_name='mobilenet_v2').get_model().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

def sliding_window_inference(images, model, window_size=(512, 512), step_size=256, device='cpu'):
    # 假设 images 已经是一个加载的 NumPy 数组
    H, W = images.shape[1], images.shape[2]
    full_output = np.zeros((H, W), dtype=np.float32)
    count_map = np.zeros((H, W), dtype=np.float32)  # 用于计算平均

    # 滑动窗口
    for i in range(0, H, step_size):
        for j in range(0, W, step_size):
            x_start = min(i, H - window_size[0])
            y_start = min(j, W - window_size[1])
            window = images[:, x_start:x_start + window_size[0], y_start:y_start + window_size[1]]
            window_tensor = torch.from_numpy(window).type(torch.float32).unsqueeze(0).to(device)

            with torch.no_grad():
                output = model(window_tensor)
                output_np = output.cpu().numpy().squeeze(0).squeeze(0)

            full_output[x_start:x_start + window_size[0], y_start:y_start + window_size[1]] += output_np
            count_map[x_start:x_start + window_size[0], y_start:y_start + window_size[1]] += 1

    # 计算平均
    full_output /= count_map

    return full_output

def save_inference(output, target_path):
    output_np = (output - output.min()) / (output.max() - output.min()) * 255
    output_img = Image.fromarray(output_np.astype(np.uint8))
    output_img.save(target_path)


def load_npy_file(npy_path):
    return np.load(npy_path).astype(np.float32)

def process_tif_for_inference(tif_path, mean, std):
    with rasterio.open(tif_path) as src:
        num_channels = src.count
        if num_channels == 12:
            channels = src.read((2, 3, 4, 8, 11, 12))  # 12通道遥感影像处理
        else:
            channels = src.read()  # 默认处理其他通道数的影像

    normalized_channels = (channels - mean[:, None, None]) / std[:, None, None]
    return normalized_channels

def inference_on_folder(source_dir, model, device, target_dir, mean, std, save_as_tiff=False, window_size=(512, 512), step_size=128):
    if not os.path.exists(target_dir):
        os.makedirs(target_dir, exist_ok=True)

    for file in tqdm(os.listdir(source_dir), desc="Processing .npy files"):
        if file.endswith('.npy'):
            npy_path = os.path.join(source_dir, file)
            images = load_npy_file(npy_path)
            output = sliding_window_inference(images, model, window_size, step_size, device)
            save_inference(output, os.path.join(target_dir, file.replace('.npy', '_inference.png')))
    
    for file in tqdm(os.listdir(source_dir), desc="Processing .tif files"):
        if file.endswith('.tif'):
            tif_path = os.path.join(source_dir, file)
            images = process_tif_for_inference(tif_path, mean, std)
            output = sliding_window_inference(images, model, window_size, step_size, device)

            if save_as_tiff:
                save_inference_as_tiff(output, tif_path, os.path.join(target_dir, file.replace('.tif', '_inference.tif')))
            else:
                save_inference(output, os.path.join(target_dir, file.replace('.tif', '_inference.png')))


if __name__ == '__main__':
    model_path = r'best.pth' # 模型权重
    source_dir = r'C:\Users\mikky\Desktop\实验结果\一张测试tif图片\测试文件夹' # 数据文件夹路径
    target_dir = r'.\data' # 生成的结果图路径
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load_model(model_path, device)
    mean = np.load('mean.npy')  # 加载标准化参数
    std = np.load('std.npy')

    save_as_tiff = False  # 默认结果保存为png格式图片，设置为True则保存tif格式的结果文件
    inference_on_folder(source_dir, model, device, target_dir, mean, std, save_as_tiff)
