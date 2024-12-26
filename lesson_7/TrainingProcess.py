import torch
from torch.utils.data import DataLoader

from lesson_7.data.PrepareDataset import load_dataset_csv

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu")
print(f"Використовується пристрій: {device}")

training_data = load_dataset_csv("training_data.csv")
test_data = load_dataset_csv("test_data.csv")

# розбиття даних на батчі
p_batch_size = 32

training_loader = DataLoader(training_data, batch_size=p_batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=p_batch_size, shuffle=False)
print(f"Датасет розбито на батчі з розміром {p_batch_size}")


