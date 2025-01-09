import numpy as np
import torch
import math
from skimage.metrics import structural_similarity


def calculate_psnr(img1, img2):
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    psnr = 20 * math.log10(1.0 / math.sqrt(mse))
    return psnr


def calculate_ssim(img1, img2):
    # Перетворюємо зображення в numpy (H, W, C)
    img1_np = img1.permute(1, 2, 0).cpu().numpy()
    img2_np = img2.permute(1, 2, 0).cpu().numpy()

    # Обчислюємо SSIM для кожного каналу
    ssim_per_channel = [
        structural_similarity(img1_np[..., c], img2_np[..., c], data_range=img2_np[..., c].max() - img2_np[..., c].min())
        for c in range(img1_np.shape[-1])
    ]

    # Повертаємо середній SSIM
    return np.mean(ssim_per_channel)
