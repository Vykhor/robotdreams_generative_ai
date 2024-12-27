import pandas as pd
import numpy as np
from torchvision import datasets
from torchvision.transforms import ToTensor, Compose, Normalize


def load_data():
    # створення трансформації: тензор + нормалізація
    transform = Compose([
        ToTensor(),
        Normalize((0.5,), (0.5,))  # нормалізація до [-1, 1]
    ])

    # завантаження даних, перетворення у тензори, нормалізація
    training_data = datasets.MNIST(
        root='./datasets',
        train=True,
        download=True,
        transform=transform
    )

    test_data = datasets.MNIST(
        root='./datasets',
        train=False,
        download=True,
        transform=transform
    )

    return training_data, test_data

def save_dataset_csv(dataset, file_path):
    images = [dataset[i][0].numpy().astype(np.float32).flatten() for i in range(len(dataset))]
    labels = [int(dataset[i][1]) for i in range(len(dataset))]
    df = pd.DataFrame(images)
    df['label'] = labels
    df.to_csv(file_path, index=False)
    print(f"Датасет збережено у файл: {file_path}")

training_data, test_data = load_data()

save_dataset_csv(training_data, "training_data.csv")
save_dataset_csv(test_data, "test_data.csv")