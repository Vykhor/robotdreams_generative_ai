import torch
import math
from torchmetrics.functional import structural_similarity_index_measure

def calculate_psnr(img1, img2):
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    psnr = 20 * math.log10(1.0 / math.sqrt(mse))
    return psnr

def calculate_ssim(img1, img2):
    ssim = structural_similarity_index_measure(img1, img2)
    return ssim.item()