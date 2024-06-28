import os
import rasterio
import numpy as np
from sklearn.model_selection import KFold
import json
from tqdm import tqdm
import argparse


def read_tiff_file(filepath):
    """
    读取TIFF文件并返回数据。
    参数:
        filepath: TIFF文件的路径。
    返回:
        数据集对象。
    """
    with rasterio.open(filepath) as dataset:
        return dataset.read()


def generate_truth_filename(scene_filename):
    """
    根据场景文件名生成对应的真值文件名。
    参数:
        scene_filename: 场景文件名。
    返回:
        真值文件名。
    """
    parts = scene_filename.split('_')
    return '_'.join(parts[:-2] + [parts[-1].replace('.tif', '_Truth.tif')])


def process_and_save_as_npz(root_dir, npz_dir, dataset_type, mean, std, out_path=None):
    """
    处理并保存为NPZ格式。
    参数:
        root_dir: 数据集的根目录。
        npz_dir: 保存NPZ文件的目录。
        dataset_type: 数据集类型（例如 'tra' 或 'val'）。
        mean: 图像数据的平均值。
        std: 图像数据的标准差。
        out_path: 输出路径。
    """
    if out_path is None:
        out_path = dataset_type
    scene_dir = os.path.join(root_dir, f'{dataset_type}_scene')
    truth_dir = os.path.join(root_dir, f'{dataset_type}_truth')
    npz_save_dir = os.path.join(npz_dir, out_path)

    if not os.path.exists(npz_save_dir):
        os.makedirs(npz_save_dir)

    print(f"处理 {dataset_type} 数据集...")
    for filename in tqdm(os.listdir(scene_dir)):
        if filename.endswith(".tif"):
            scene_filepath = os.path.join(scene_dir, filename)
            truth_filename = generate_truth_filename(filename)
            truth_filepath = os.path.join(truth_dir, truth_filename)

            scene_data = read_tiff_file(scene_filepath)
            truth_data = read_tiff_file(truth_filepath)

            # 应用归一化
            normalized_scene = (
                scene_data - mean[:, None, None]) / std[:, None, None]

            npz_filename = os.path.splitext(filename)[0] + '.npz'
            np.savez_compressed(os.path.join(
                npz_save_dir, npz_filename), images=normalized_scene, label=truth_data)


# def process_file(npz_file_path, target_dir):
#     """
#     处理单个NPZ文件并保存切片。
#     参数:
#         npz_file_path: NPZ文件的路径。
#         target_dir: 切片保存的目标目录。
#     """
#     with np.load(npz_file_path) as data:
#         images = data['images']
#         label = data['label']

#     os.makedirs(target_dir, exist_ok=True)

#     num_slices_x = images.shape[1] // 256
#     num_slices_y = images.shape[2] // 256

#     for i in range(num_slices_x + 1):
#         for j in range(num_slices_y + 1):
#             x_start = i * 256
#             y_start = j * 256

#             if x_start + 256 > images.shape[1]:
#                 x_start = images.shape[1] - 256
#             if y_start + 256 > images.shape[2]:
#                 y_start = images.shape[2] - 256

#             images_slice = images[:, x_start:x_start+256, y_start:y_start+256]
#             label_slice = label[:, x_start:x_start+256, y_start:y_start+256]

#             slice_filename = f"{target_dir}/slice_{x_start}_{y_start}.npz"
#             np.savez(slice_filename, images=images_slice, label=label_slice)

def process_file(npz_file_path, target_dir, slice_size=512):
    """
    处理单个NPZ文件并保存切片。
    参数:
        npz_file_path: NPZ文件的路径。
        target_dir: 切片保存的目标目录。
        slice_size: 切片的大小，默认为256。
    """
    with np.load(npz_file_path) as data:
        images = data['images']
        label = data['label']

    os.makedirs(target_dir, exist_ok=True)

    num_slices_x = images.shape[1] // slice_size
    num_slices_y = images.shape[2] // slice_size

    for i in range(num_slices_x + 1):
        for j in range(num_slices_y + 1):
            x_start = i * slice_size
            y_start = j * slice_size

            if x_start + slice_size > images.shape[1]:
                x_start = images.shape[1] - slice_size
            if y_start + slice_size > images.shape[2]:
                y_start = images.shape[2] - slice_size

            images_slice = images[:, x_start:x_start +
                                  slice_size, y_start:y_start+slice_size]
            label_slice = label[:, x_start:x_start +
                                slice_size, y_start:y_start+slice_size]

            slice_filename = f"{target_dir}/slice_{x_start}_{y_start}.npz"
            np.savez(slice_filename, images=images_slice, label=label_slice)


def perform_kfold_split(root_dir):
    """
    对指定目录下的文件进行五折交叉验证划分。
    参数:
        root_dir: 包含数据片段的根目录。
    """
    directories = [d for d in os.listdir(
        root_dir) if os.path.isdir(os.path.join(root_dir, d))]

    kf = KFold(n_splits=5)
    folds = {}

    for i, (train_index, test_index) in enumerate(tqdm(kf.split(directories), desc="五折划分进度")):
        train_dirs = [directories[idx] for idx in train_index]
        test_dirs = [directories[idx] for idx in test_index]
        folds[f'fold_{i+1}'] = {'train': train_dirs, 'val': test_dirs}

    json_path = os.path.join(root_dir, 'folds.json')
    with open(json_path, 'w') as f:
        json.dump(folds, f, indent=4)

    print(f"五折分割信息已保存至 {json_path}")


def main(root_dir, npz_dir):
    """
    主程序函数。
    """
    # 先计算训练集图像的平均值和标准差
    mean, std = calculate_mean_std(root_dir, 'tra')
    np.save(os.path.join(npz_dir, 'mean.npy'), mean)
    np.save(os.path.join(npz_dir, 'std.npy'), std)

    # 处理并保存训练集和验证集为 NPZ 格式
    process_and_save_as_npz(root_dir, npz_dir, 'tra', mean, std)
    process_and_save_as_npz(root_dir, npz_dir, 'val',
                            mean, std, out_path='test')

    train_root = os.path.join(npz_dir, 'tra')
    train_target_root = os.path.join(npz_dir, 'train_pieces')
    npz_files = [f for f in os.listdir(train_root) if f.endswith('.npz')]

    print("开始切分数据片段...")
    for file in tqdm(npz_files, desc="文件处理进度"):
        npz_file_path = os.path.join(train_root, file)
        target_dir = os.path.join(train_target_root, os.path.splitext(file)[0])
        process_file(npz_file_path, target_dir)

    perform_kfold_split(train_target_root)


def calculate_mean_std(root_dir, dataset_type):
    """
    计算给定数据集的图像的平均值和标准差。
    参数:
        root_dir: 数据集的根目录。
        dataset_type: 数据集类型（例如 'train' 或 'val'）。
    """
    scene_dir = os.path.join(root_dir, f'{dataset_type}_scene')
    mean_sum = np.zeros(6)  # 假设有6个波段
    std_sum = np.zeros(6)
    num_images = 0

    for filename in os.listdir(scene_dir):
        if filename.endswith(".tif"):
            scene_filepath = os.path.join(scene_dir, filename)
            with rasterio.open(scene_filepath) as dataset:
                image = dataset.read()
                # if np.any(image < 0) or np.any(np.isnan(image)):
                #     continue
                mean_sum += np.mean(image, axis=(1, 2))
                std_sum += np.std(image, axis=(1, 2))
                num_images += 1

    mean = mean_sum / num_images
    std = std_sum / num_images

    return mean, std


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='处理TIFF文件并生成NPZ文件。')
    parser.add_argument('-r', '--root_dir', required=True, help='数据集的根目录。')
    parser.add_argument('-o', '--output_dir',
                        required=True, help='输出NPZ文件的目录。')

    args = parser.parse_args()

    main(args.root_dir, args.output_dir)


# python preprocess_files.py -r "F:\需要认真总结个项目了\敲代码\data\dset-s2" -o "F:\需要认真总结个项目了\敲代码\data\npz"
