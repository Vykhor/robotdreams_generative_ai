import random
import torch

def add_noise(images, noise_level=0.1):
    noise = torch.randn_like(images) * noise_level
    return torch.clamp(images + noise, 0, 1)

def apply_blur(images, kernel_size=3):
    return torch.nn.functional.avg_pool2d(images, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)

def add_occlusion(images, size=8):
    for img in images:
        x = random.randint(0, img.size(1) - size)
        y = random.randint(0, img.size(2) - size)
        img[:, x:x+size, y:y+size] = 0
    return images

def apply_pixelation(images, scale=4):
    downscaled = torch.nn.functional.interpolate(images, scale_factor=1/scale, mode='bilinear')
    return torch.nn.functional.interpolate(downscaled, size=images.shape[2:], mode='bilinear')

# Функція для пошкодження зображень
def corrupt_images(images):
    corrupted_images = images.clone()
    for i in range(corrupted_images.size(0)):
        method = random.choice(["noise", "blur", "occlusion", "pixelation"])
        if method == "noise":
            corrupted_images[i] = add_noise(corrupted_images[i].unsqueeze(0)).squeeze(0)
        elif method == "blur":
            corrupted_images[i] = apply_blur(corrupted_images[i].unsqueeze(0)).squeeze(0)
        elif method == "occlusion":
            corrupted_images[i] = add_occlusion(corrupted_images[i].unsqueeze(0)).squeeze(0)
        elif method == "pixelation":
            corrupted_images[i] = apply_pixelation(corrupted_images[i].unsqueeze(0)).squeeze(0)
    return corrupted_images