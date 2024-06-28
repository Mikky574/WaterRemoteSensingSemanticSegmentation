import matplotlib.pyplot as plt
import os

def calculate_dice(pred, target, smooth=1.0):
    pred_flat = pred.view(-1)
    target_flat = target.view(-1)
    intersection = (pred_flat * target_flat).sum()
    return (2.0 * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)

def update_ema(ema_dice, new_dice, alpha=0.9):
    return alpha * ema_dice + (1 - alpha) * new_dice

def load_cloud_images(cloud_root):
    """
    从指定目录加载所有云图像。

    Args:
    cloud_root (str): 云图像存储的根目录。

    Returns:
    list of np.ndarray: 加载的云图像列表。
    """
    cloud_images = []
    for filename in os.listdir(cloud_root):
        if filename.endswith('.npy'):
            cloud_path = os.path.join(cloud_root, filename)
            cloud_image = np.load(cloud_path)
            cloud_images.append(cloud_image)
    return cloud_images

