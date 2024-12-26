import torch
from torch.utils.data import DataLoader

from lesson_7.data.Dataset import load_dataset_csv
from lesson_7.model.Discriminator import create_discriminator
from lesson_7.model.Generator import create_generator

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

# параметри
latent_dim = 100
image_size = 28*28
negative_slope = 0.2
dropout = 0.3

mnist_generator = create_generator(latent_dim, image_size)
mnist_discriminator = create_discriminator(image_size, negative_slope, dropout)

