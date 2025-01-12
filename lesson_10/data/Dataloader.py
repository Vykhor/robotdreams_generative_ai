import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader


def load_dataset_csv(file_path):
    df = pd.read_csv(file_path)
    # Завантажуємо всі зображення без обробки стовпця 'label'
    images = torch.tensor(df.values, dtype=torch.float32).view(-1, 3, 32, 32)
    print(f"Датасет завантажено з файлу: {file_path}")
    return TensorDataset(images)

def get_dataloader(file_path, batch_size, shuffle):
    dataset = load_dataset_csv(file_path)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)