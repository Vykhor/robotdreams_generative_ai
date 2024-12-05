import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor
from torch.utils.data import TensorDataset
import pandas as pd
import numpy as np

device = torch.device(
    "cuda" if torch.cuda.is_available() 
    else "xpu" if torch.xpu.is_available()
    else "cpu")
print(f"Використовується пристрій: {device}")

# аугментація
train_transforms = transforms.Compose([
    transforms.RandomRotation(10),
    transforms.RandomAffine(10, translate=(0.1, 0.1)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
])

# завантаження даних, перетворення у тензори, нормалізація
training_data = datasets.MNIST(
    root='./datasets',
    train=True,
    download=True,
    transform=ToTensor()
)

augmented_data = datasets.MNIST(
    root='./datasets',
    train=True,
    download=True,
    transform=train_transforms
)

test_data = datasets.MNIST(
    root='./datasets',
    train=False,
    download=True,
    transform=ToTensor()
)

# перевірка та фільтрація датасету
def filter_missing_values(data):
    valid_indices = []
    for i, (image, label) in enumerate(data):
        if not torch.isnan(image).any() and label is not None:
            valid_indices.append(i)
        else:
            print(f"Пропущені значення в зображенні {i}, видалено.")
    return torch.utils.data.Subset(data, valid_indices)

def filter_anomalies(data):
    valid_indices = []
    for i, (image, label) in enumerate(data):
        if 0 <= image.min() <= image.max() <= 1 and image.size() == (1, 28, 28):
            valid_indices.append(i)
        else:
            print(f"Аномалія у зображенні {i}: розмір {image.size()} або значення пікселів, видалено.")
    return torch.utils.data.Subset(data, valid_indices)

def filter_data_types(data):
    valid_indices = []
    for i, (image, label) in enumerate(data):
        if isinstance(image, torch.Tensor) and (isinstance(label, int) or isinstance(label, torch.Tensor)):
            if isinstance(label, torch.Tensor):  # якщо мітка є тензором (виникає після аугментації)
                label = int(label.item())  # конвертація в int
            valid_indices.append(i)
        else:
            print(f"Некоректні типи даних у зображенні {i}: тип {type(image)} або мітки {type(label)}, видалено.")
    return torch.utils.data.Subset(data, valid_indices)

def subset_to_dataset(subset):
    images = []
    labels = []
    for i in range(len(subset)):
        image, label = subset[i]
        images.append(image)
        labels.append(label)
    images_tensor = torch.stack(images)  # Розмір: [N, 1, 28, 28]
    labels_tensor = torch.tensor(labels)  # Розмір: [N]
    return TensorDataset(images_tensor, labels_tensor)

print("Перевірка training_data")
training_data = filter_missing_values(training_data)
training_data = filter_anomalies(training_data)
training_data = filter_data_types(training_data)
training_data = subset_to_dataset(training_data)

print("Перевірка augmented_data")
training_data = filter_missing_values(training_data)
training_data = filter_anomalies(training_data)
training_data = filter_data_types(training_data)
training_data = subset_to_dataset(training_data)

print("Перевірка test_data")
test_data = filter_missing_values(test_data)
test_data = filter_anomalies(test_data)
test_data = filter_data_types(test_data)
test_data = subset_to_dataset(test_data)

print(f"Кількість зображень у training_data після перевірки: {len(training_data)}")
print(f"Кількість зображень у augmented_data після перевірки: {len(augmented_data)}")
print(f"Кількість зображень у test_data після перевірки: {len(test_data)}")

# збереження відфільтрованого dataset в CSV
def save_dataset_csv(dataset, file_path):
    images = [dataset[i][0].numpy().astype(np.float32).flatten() for i in range(len(dataset))]
    labels = [int(dataset[i][1]) for i in range(len(dataset))]
    df = pd.DataFrame(images)
    df['label'] = labels
    df.to_csv(file_path, index=False)
    print(f"Датасет збережено у файл: {file_path}")

# збереження двох відфільтрованих dataset в CSV
def save_datasets_csv(first_dataset, second_dataset, file_path):
    first_images = [first_dataset[i][0].numpy().astype(np.float32).flatten() for i in range(len(first_dataset))]
    first_labels = [int(first_dataset[i][1]) for i in range(len(first_dataset))]

    second_images = [second_dataset[i][0].numpy().astype(np.float32).flatten() for i in range(len(second_dataset))]
    second_labels = [int(second_dataset[i][1]) for i in range(len(second_dataset))]

    images = first_images + second_images
    labels = first_labels + second_labels

    df = pd.DataFrame(images)
    df['label'] = labels
    df.to_csv(file_path, index=False)
    print(f"Обидва датасети збережено у файл: {file_path}")

save_datasets_csv(training_data, augmented_data, "../datasets/training_data.csv")
save_dataset_csv(test_data, "../datasets/test_data.csv")
