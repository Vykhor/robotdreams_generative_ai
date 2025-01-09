import pandas as pd
import torch
from datetime import datetime

from torch.utils.data import DataLoader, TensorDataset
from torch import nn, optim

from lesson_9.Vizualization import plot, plot_comparing
from lesson_9.model.Generator import create_generator
from lesson_9.model.Discriminator import create_discriminator
from lesson_9.Metrics import calculate_psnr
from lesson_9.Metrics import calculate_ssim

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu")
print(f"Використовується пристрій: {device}")

def load_dataset_csv(file_path):
    df = pd.read_csv(file_path)
    # Завантажуємо лише зображення
    images = torch.tensor(df.drop(columns=['label']).values, dtype=torch.float32).view(-1, 3, 32, 32)
    print(f"Датасет завантажено з файлу: {file_path}")
    return TensorDataset(images)

# Використання DataLoader
def get_dataloader(file_path, batch_size, shuffle=True):
    dataset = load_dataset_csv(file_path)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

# Параметри завантаження даних
file_path = "./cifar10_training_data.csv"
p_batch_size = 32

# Створення DataLoader
training_loader = get_dataloader(file_path, p_batch_size)
num_batches = len(training_loader)
print(f"Датасет розбито на {num_batches} батчів розміром {p_batch_size}.")

# Параметри моделей
p_negative_slope = 0.2

# Ініціалізація моделей
generator = create_generator(p_negative_slope).to(device)
discriminator = create_discriminator(p_negative_slope).to(device)

# Параметри навчання
num_epochs = 20
lr_gen = 0.0002
lr_disc = 0.0001
beta1, beta2 = 0.5, 0.999

# Функції втрат
discriminator_loss_function = nn.BCELoss()
generator_loss_function = nn.L1Loss()

# Оптимізатори
optimizer_g = optim.Adam(generator.parameters(), lr=lr_gen, betas=(beta1, beta2))
optimizer_d = optim.Adam(discriminator.parameters(), lr=lr_disc, betas=(beta1, beta2))

current_time = datetime.now().strftime("%H:%M:%S")
print(f"[{current_time}] Початок процесу навчання")

psnr_list = []
ssim_list = []

for epoch in range(num_epochs):
    for batch in training_loader:

        damaged_images = batch[0].to(device)
        real_images = damaged_images.clone()

        # --- Крок 1: Оновлення дискримінатора ---
        optimizer_d.zero_grad()

        real_labels = torch.ones(damaged_images.size(0), 1).to(device)
        fake_labels = torch.zeros(damaged_images.size(0), 1).to(device)

        output_real = discriminator(real_images)
        loss_real = discriminator_loss_function(output_real, real_labels)

        fake_images = generator(damaged_images)
        output_fake = discriminator(fake_images.detach())
        loss_fake = discriminator_loss_function(output_fake, fake_labels)

        loss_d = loss_real + loss_fake
        loss_d.backward()
        optimizer_d.step()

        # --- Крок 2: Оновлення генератора ---
        optimizer_g.zero_grad()

        output_fake_for_g = discriminator(fake_images)
        loss_g_adv = discriminator_loss_function(output_fake_for_g, real_labels)

        loss_g_rec = generator_loss_function(fake_images, real_images)

        loss_g = loss_g_adv + 100 * loss_g_rec
        loss_g.backward()
        optimizer_g.step()

    current_time = datetime.now().strftime("%H:%M:%S")
    print(f"[{current_time}] | Епоха [{epoch + 1}/{num_epochs}] завершена. Обчислення метрик.")

    # Обчислення метрик після кожної епохи
    psnr_total = 0
    ssim_total = 0

    for batch in training_loader:
        damaged_images = batch[0].to(device)
        real_images = damaged_images.clone()

        with torch.no_grad():
            restored_images = generator(damaged_images)
            psnr_total += calculate_psnr(restored_images, real_images)
            ssim_total += calculate_ssim(restored_images, real_images)

    avg_psnr = psnr_total / num_batches
    avg_ssim = ssim_total / num_batches

    # Додаємо середні значення метрик до списків
    psnr_list.append(avg_psnr)
    ssim_list.append(avg_ssim)

    current_time = datetime.now().strftime("%H:%M:%S")
    print(f"[{current_time}] PSNR: {avg_psnr:.2f}  SSIM: {avg_ssim:.4f}")

    # Візуалізація порівняння пошкодженого та відновленого зображення
    if epoch % 1 == 0:  # Наприклад, кожну епоху
        plot_comparing(damaged_image=damaged_images[0], restored_image=restored_images[0], real_image=real_images[0])

plot(num_epochs=num_epochs, psnr_list=psnr_list,  ssim_list=ssim_list)