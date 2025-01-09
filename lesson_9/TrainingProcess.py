import pandas as pd
import torch
from datetime import datetime
from torch.utils.data import DataLoader, TensorDataset
from torch import nn, optim

from lesson_9.model.Generator import create_generator
from lesson_9.model.Discriminator import create_discriminator

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu")
print(f"Використовується пристрій: {device}")

def load_dataset_csv(file_path):
    df = pd.read_csv(file_path)
    labels = torch.tensor(df['label'].values, dtype=torch.long)
    images = torch.tensor(df.drop(columns=['label']).values, dtype=torch.float32).view(-1, 1, 28, 28)
    print(f"Датасет завантажено з файлу: {file_path}")
    return TensorDataset(images, labels)

training_data = load_dataset_csv("../training_data.csv")
print(f"Розмір навчального набору: {len(training_data)}")

# розбиття даних на батчі
p_batch_size = 32

training_loader = DataLoader(training_data, batch_size=p_batch_size, shuffle=True)

print(f"Датасет розбито на батчі з розміром {p_batch_size}")

# Ініціалізація моделей
generator = create_generator(0.2)
discriminator = create_discriminator(0.2)

# Функції втрат
adversarial_loss = nn.BCELoss()  # Для дискримінатора
reconstruction_loss = nn.L1Loss()  # Для генератора

# Оптимізатори
lr_gen = 0.0002
lr_disc = 0.0001
beta1, beta2 = 0.5, 0.999
optimizer_g = optim.Adam(generator.parameters(), lr=lr_gen, betas=(beta1, beta2))
optimizer_d = optim.Adam(discriminator.parameters(), lr=lr_disc, betas=(beta1, beta2))

# Параметри навчання
num_epochs = 50
batch_size = 64

# Приклад даних для тренування
real_images = torch.randn(batch_size, 3, 64, 64)  # Реальні зображення
damaged_images = torch.randn(batch_size, 3, 64, 64)  # Пошкоджені зображення

for epoch in range(num_epochs):
    # --- Крок 1: Оновлення дискримінатора ---
    optimizer_d.zero_grad()

    # Вихід дискримінатора для реальних зображень
    real_labels = torch.ones(batch_size, 1)
    output_real = discriminator(real_images)
    loss_real = adversarial_loss(output_real, real_labels)

    # Вихід дискримінатора для згенерованих зображень
    fake_images = generator(damaged_images)
    fake_labels = torch.zeros(batch_size, 1)
    output_fake = discriminator(fake_images.detach())
    loss_fake = adversarial_loss(output_fake, fake_labels)

    # Загальна втрата дискримінатора
    loss_d = loss_real + loss_fake
    loss_d.backward()
    optimizer_d.step()

    # --- Крок 2: Оновлення генератора ---
    optimizer_g.zero_grad()

    # Adversarial loss для генератора
    output_fake_for_g = discriminator(fake_images)
    loss_g_adv = adversarial_loss(output_fake_for_g, real_labels)

    # Reconstruction loss
    loss_g_rec = reconstruction_loss(fake_images, real_images)

    # Загальна втрата генератора
    loss_g = loss_g_adv + 100 * loss_g_rec
    loss_g.backward()
    optimizer_g.step()

    # Логування втрат
    if (epoch + 1) % 10 == 0:
        current_time = datetime.now().strftime("%H:%M:%S")
        print(f"{current_time} | Epoch [{epoch + 1}/{num_epochs}]  Loss_D: {loss_d.item():.4f}  Loss_G: {loss_g.item():.4f}")