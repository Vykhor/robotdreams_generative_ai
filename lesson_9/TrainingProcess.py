import torch
from datetime import datetime
from torch import nn, optim

from lesson_9.data.Dataloader import get_dataloader
from lesson_9.plot.Plotter import plot, plot_comparing
from lesson_9.model.Generator import create_generator
from lesson_9.model.Discriminator import create_discriminator
from lesson_9.Metrics import calculate_metrics_on_batch

from lesson_9.data.Corrupting import corrupt_images

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu")
print(f"Використовується пристрій: {device}")

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

def weights_init(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.xavier_uniform_(m.weight)  # Ініціалізація для згорткових шарів
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.ones_(m.weight)  # Ініціалізація gamma у BatchNorm
        nn.init.zeros_(m.bias)  # Ініціалізація beta
    elif isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)  # Ініціалізація для лінійних шарів
        nn.init.zeros_(m.bias)

generator.apply(weights_init)
discriminator.apply(weights_init)

# Параметри навчання
num_epochs = 20
lr_gen = 0.0002
lr_disc = 0.000005
beta1, beta2 = 0.5, 0.999
p_generator_iterations = 2

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

discriminator.train()
generator.train()
for epoch in range(num_epochs):

    total_loss_g = 0  # Для накопичення втрат генератора
    total_loss_d = 0  # Для накопичення втрат дискримінатора
    num_batches = len(train_loader)

    for i, (batch) in enumerate(train_loader):
        real_images = batch[0].to(device)
        damaged_images = corrupt_images(real_images)
        batch_size = real_images.size(0)

        if i % 2 == 0:
            # --- Крок 1: Оновлення дискримінатора ---
            optimizer_d.zero_grad()

            real_labels = torch.ones(batch_size, 1).to(device) * 0.9 + torch.rand(batch_size, 1).to(device) * 0.1
            fake_labels = torch.zeros(batch_size, 1).to(device) + torch.rand(batch_size, 1).to(device) * 0.1

            output_real = discriminator(real_images)
            loss_real = discriminator_loss_function(output_real, real_labels)

            fake_images = generator(damaged_images)

            output_fake = discriminator(fake_images.detach())
            loss_fake = discriminator_loss_function(output_fake, fake_labels)

            loss_d = loss_real + loss_fake
            loss_d.backward()
            optimizer_d.step()

            # Накопичення втрати дискримінатора
            total_loss_d += loss_d.item()

        # --- Крок 2: Оновлення генератора ---
        for _ in range(p_generator_iterations):
            real_labels = torch.ones(batch_size, 1).to(device)
            fake_labels = torch.zeros(batch_size, 1).to(device)

            optimizer_g.zero_grad()

            fake_images = generator(damaged_images)

            output_fake_for_g = discriminator(fake_images)
            loss_g_adv = discriminator_loss_function(output_fake_for_g, real_labels)
            loss_g_rec = generator_loss_function(fake_images, real_images)

            loss_g = loss_g_adv + 50 * loss_g_rec
            loss_g.backward()
            optimizer_g.step()

            # Накопичення втрати генератора
            total_loss_g += loss_g.item()

    # Розрахунок середніх втрат після епохи
    avg_loss_g = total_loss_g / (num_batches * p_generator_iterations)
    avg_loss_d = total_loss_d / (num_batches / 2)

    current_time = datetime.now().strftime("%H:%M:%S")
    print(f"[{current_time}] | Епоха [{epoch + 1}/{num_epochs}] завершена. Середня втрата генератора: {avg_loss_g:.4f}, дискримінатора: {avg_loss_d:.4f}.")

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
torch.save(generator.state_dict(), 'result/generator.pth')
torch.save(discriminator.state_dict(), 'result/discriminator.pth')
print("Моделі збережено у файли: 'generator.pth' та 'discriminator.pth'")

plot(num_epochs=num_epochs, psnr_list=psnr_list,  ssim_list=ssim_list)