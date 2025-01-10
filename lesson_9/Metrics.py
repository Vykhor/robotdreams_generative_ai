import numpy as np
import torch
from skimage.metrics import structural_similarity


def calculate_psnr_batch(img1, img2):
    # Обчислення MSE для кожного зображення в батчі
    mse = torch.mean((img1 - img2) ** 2, dim=(1, 2, 3))  # Обчислюємо по всіх пікселях для кожного зображення
    psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))  # Використовуємо логарифм для отримання PSNR
    psnr[mse == 0] = float('inf')  # Якщо MSE == 0, то PSNR = нескінченність
    return psnr


def calculate_ssim_batch(img1, img2):
    batch_size = img1.shape[0]
    ssim_values = []

    for i in range(batch_size):
        # Перетворюємо зображення в numpy (H, W, C)
        img1_np = img1[i].permute(1, 2, 0).cpu().numpy()
        img2_np = img2[i].permute(1, 2, 0).cpu().numpy()

        # Обчислюємо SSIM для кожного каналу
        ssim_per_channel = [
            structural_similarity(img1_np[..., c], img2_np[..., c],
                                  data_range=img2_np[..., c].max() - img2_np[..., c].min())
            for c in range(img1_np.shape[-1])
        ]

        # Зберігаємо середній SSIM для поточного зображення
        ssim_values.append(np.mean(ssim_per_channel))

    return np.mean(ssim_values)


def calculate_metrics_on_batch(damaged_images, real_images, generator):
    with torch.no_grad():
        restored_images = generator(damaged_images)

        # Обчислення PSNR та SSIM для кожного батча
        psnr_batch = calculate_psnr_batch(restored_images, real_images)  # Повертає вектор з PSNR для кожного зображення
        ssim_batch = calculate_ssim_batch(restored_images, real_images)  # Середнє SSIM для батча

        # Сума всіх значень для обчислення середнього
        psnr_total = psnr_batch.mean().item()  # Середнє значення PSNR для батчу
        ssim_total = ssim_batch  # Середнє SSIM для батчу

    return psnr_total, ssim_total