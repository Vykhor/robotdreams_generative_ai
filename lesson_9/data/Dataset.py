import pandas as pd
from torchvision import datasets, transforms
from torchvision.datasets import CIFAR10
from torchvision.transforms import Normalize

def load_cifar10_data():
    # Трансформації (перетворення) для зображень
    transform = transforms.Compose([
        transforms.ToTensor(),  # Перетворення до тензора
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Нормалізація
    ])

    # Завантаження тренувального і тестового наборів
    train_dataset = CIFAR10(root="./data", train=True, download=True, transform=transform)
    test_dataset = CIFAR10(root="./data", train=False, download=True, transform=transform)
    return train_dataset, test_dataset

def load_celeba_data():
    # Трансформації для обробки зображень CelebA
    image_transforms = transforms.Compose([
        transforms.Resize((64, 64)),  # Змінюємо розмір зображень до 64x64
        transforms.ToTensor(),
        Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Нормалізація для RGB
    ])
    celeba_dataset = datasets.CelebA(root='./datasets', split='train', download=True, transform=image_transforms)
    return celeba_dataset

def save_dataset_csv(dataset, file_path):
    # Отримуємо зображення з датасету та перетворюємо їх у вектори
    images = [image.flatten().tolist() for image, _ in dataset]

    # Створюємо DataFrame
    df = pd.DataFrame(images)

    # Зберігаємо у CSV
    df.to_csv(file_path, index=False)
    print(f"Зображення збережено у файл: {file_path}")

# Завантаження та збереження датасетівв
#celeba_training_data = load_celeba_data()
#save_dataset_csv(celeba_training_data, "../celeba_training_data.csv")

test_data, training_data  = load_cifar10_data()
save_dataset_csv(training_data, "../cifar10_training_data.csv")
save_dataset_csv(test_data, "../cifar10_test_data.csv")
