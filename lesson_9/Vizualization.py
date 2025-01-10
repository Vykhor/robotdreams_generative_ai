import torch
from matplotlib import pyplot as plt

# Побудова графіків змін метрик
def plot(num_epochs, psnr_list, ssim_list):
    plt.figure(figsize=(12, 6))

    # Графік для PSNR
    plt.subplot(1, 2, 1)
    plt.plot(range(1, num_epochs + 1), psnr_list, label='PSNR', color='blue', marker='o')
    plt.title('Зміна PSNR по епохах')
    plt.xlabel('Епоха')
    plt.ylabel('PSNR')
    plt.grid(True)

    # Графік для SSIM
    plt.subplot(1, 2, 2)
    plt.plot(range(1, num_epochs + 1), ssim_list, label='SSIM', color='red', marker='o')
    plt.title('Зміна SSIM по епохах')
    plt.xlabel('Епоха')
    plt.ylabel('SSIM')
    plt.grid(True)

    plt.tight_layout()
    plt.show()

def plot_comparing(damaged_image, restored_image, real_image):
    plt.figure(figsize=(12, 6))

    # Нормалізація зображень (якщо вони мають значення поза діапазоном [0, 1])
    damaged_image = torch.clamp(damaged_image, 0, 1)
    restored_image = torch.clamp(restored_image, 0, 1)
    real_image = torch.clamp(real_image, 0, 1)

    # Порівняння зображень
    plt.subplot(1, 3, 1)
    plt.imshow(damaged_image.cpu().permute(1, 2, 0).detach().numpy())
    plt.title("Пошкоджене зображення")
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(restored_image.cpu().permute(1, 2, 0).detach().numpy())
    plt.title("Відновлене зображення")
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(real_image.cpu().permute(1, 2, 0).detach().numpy())
    plt.title("Оригінальне зображення")
    plt.axis('off')

    plt.tight_layout()
    plt.show()