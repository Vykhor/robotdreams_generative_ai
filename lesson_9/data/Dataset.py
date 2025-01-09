import pandas as pd
import numpy as np
import torch
from torchvision import datasets, transforms
from torchvision.transforms import Normalize
import cv2
import random

def load_celeba_data():
    # Трансформації для обробки зображень CelebA
    image_transforms = transforms.Compose([
        transforms.Resize((64, 64)),  # Змінюємо розмір зображень до 64x64
        transforms.ToTensor(),
        Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Нормалізація для RGB
    ])

    # Завантаження датасету CelebA
    celeba_dataset = datasets.CelebA(
        root='./datasets',
        split='train',
        download=True,
        transform=image_transforms
    )

    return celeba_dataset

# Розмиття
def apply_blur(image, kernel_size=(5, 5)):
    # Перетворення тензора у формат OpenCV
    img_array = np.transpose(image.numpy(), (1, 2, 0))  # [C, H, W] -> [H, W, C]
    img_array = (img_array * 255).astype(np.uint8)  # Повертаємо до [0, 255]

    # Гаусове розмиття
    blurred = cv2.GaussianBlur(img_array, kernel_size, 0)

    # Перетворення назад у тензор
    blurred_tensor = torch.tensor(np.transpose(blurred / 255.0, (2, 0, 1)),
                                  dtype=torch.float32)  # [H, W, C] -> [C, H, W]
    return blurred_tensor

# Шум
def add_gaussian_noise(image, mean=0, std=0.1):
    noise = torch.randn(image.size()) * std + mean
    noisy_image = image + noise
    return torch.clip(noisy_image, -1, 1)  # Обмеження значень між [-1, 1]

# Випадкове пошкодження зображень
def apply_random_corruption(image):
    corruption_type = random.choice(["blur", "noise"])
    if corruption_type == "blur":
        return apply_blur(image)
    elif corruption_type == "noise":
        return add_gaussian_noise(image)
    else:
        return image

def save_celeba_dataset_csv(dataset, file_path):
    images = []
    labels = []

    for i in range(len(dataset)):
        image, label = dataset[i]
        corrupted_image = apply_random_corruption(image)
        images.append(corrupted_image.numpy().astype(np.float32).flatten())
        labels.append(label)

    # Перетворюємо на DataFrame
    df = pd.DataFrame(images)
    df['label'] = labels

    # Збереження у CSV
    df.to_csv(file_path, index=False)
    print(f"Датасет CelebA збережено у файл: {file_path}")

# Завантаження датасету
training_data = load_celeba_data()
# Збереження у файл
save_celeba_dataset_csv(training_data, "../celeba_training_data.csv")