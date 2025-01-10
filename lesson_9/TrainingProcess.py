import pandas as pd
import torch
from datetime import datetime

from torch.utils.data import DataLoader, TensorDataset
from torch import nn, optim

from lesson_9.Vizualization import plot, plot_comparing
from lesson_9.model.Generator import create_generator
from lesson_9.model.Discriminator import create_discriminator
from lesson_9.Metrics import calculate_metrics_on_batch

from lesson_9.data.Corrupting import corrupt_images

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu")
print(f"Використовується пристрій: {device}")

def load_dataset_csv(file_path):
    df = pd.read_csv(file_path)
    # Завантажуємо всі зображення без обробки стовпця 'label'
    images = torch.tensor(df.values, dtype=torch.float32).view(-1, 3, 32, 32)
    print(f"Датасет завантажено з файлу: {file_path}")
    return TensorDataset(images)


def get_dataloader(file_path, batch_size, shuffle):
    dataset = load_dataset_csv(file_path)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

# Параметри завантаження даних
training_file_path = "./cifar10_training_data.csv"
test_file_path = "./cifar10_test_data.csv"
p_batch_size = 32

# Створення DataLoader для тренувальних і тестових даних
train_loader = get_dataloader(training_file_path, p_batch_size, shuffle=True)
test_loader = get_dataloader(test_file_path, p_batch_size, shuffle=False)

# Перевірка розміру наборів
print(f"Тренувальний набір: {len(train_loader)} батчів")
print(f"Тестовий набір: {len(test_loader)} батчів")

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

psnr_list = []
ssim_list = []
damaged_test_images = []

for batch in test_loader:
    real_images = batch[0].to(device)
    with torch.no_grad():
        damaged_images = corrupt_images(real_images)
    damaged_test_images.append(damaged_images)

# Об’єднуємо всі батчі в один тензор
damaged_test_images = torch.cat(damaged_test_images, dim=0)
print(f"Пошкоджені тестові зображення створені: {damaged_test_images.shape}")

current_time = datetime.now().strftime("%H:%M:%S")
print(f"[{current_time}] Початок процесу навчання")
for epoch in range(num_epochs):
    for batch in train_loader:
        real_images = batch[0].to(device)
        damaged_images = corrupt_images(real_images)

        # --- Крок 1: Оновлення дискримінатора ---
        optimizer_d.zero_grad()

        real_labels = torch.ones(damaged_images.size(0), 1).to(device)
        fake_labels = torch.zeros(damaged_images.size(0), 1).to(device)

        output_real = discriminator(real_images)
        loss_real = discriminator_loss_function(output_real, real_labels)

        fake_images = generator(damaged_images)
        fake_images = (fake_images + 1) / 2 # Нормалізація

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
    num_batches = len(test_loader)

    # Обчислення метрик для всіх батчів тестового набору
    for i, batch in enumerate(test_loader):
        damaged_images = damaged_test_images[i * p_batch_size: (i + 1) * p_batch_size].to(device)
        real_images = batch[0].to(device)

        psnr_batch, ssim_batch = calculate_metrics_on_batch(damaged_images, real_images, generator)
        psnr_total += psnr_batch
        ssim_total += ssim_batch

    avg_psnr = psnr_total / num_batches
    avg_ssim = ssim_total / num_batches

    # Додаємо середні значення метрик до списків
    psnr_list.append(avg_psnr)
    ssim_list.append(avg_ssim)

    current_time = datetime.now().strftime("%H:%M:%S")
    print(f"[{current_time}] PSNR: {avg_psnr:.2f}  SSIM: {avg_ssim:.4f}")

    # Візуалізація порівняння пошкодженого та відновленого зображення
    #if epoch == (num_epochs - 1):
    #   plot_comparing(damaged_image=damaged_images[0], restored_image=fake_images[0], real_image=real_images[0])

# Збереження натренованих моделей
torch.save(generator.state_dict(), 'generator.pth')
torch.save(discriminator.state_dict(), 'discriminator.pth')
print("Моделі збережено у файли: 'generator.pth' та 'discriminator.pth'")

plot(num_epochs=num_epochs, psnr_list=psnr_list,  ssim_list=ssim_list)