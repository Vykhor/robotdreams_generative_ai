import pandas as pd
import torch
from datetime import datetime
from torch.utils.data import DataLoader, TensorDataset
from torch import nn

from lesson_8.congan.model.Discriminator import create_discriminator
from lesson_8.congan.model.Generator import create_generator

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

training_data = load_dataset_csv("../common/data/training_data.csv")
print(f"Розмір навчального набору: {len(training_data)}")

# розбиття даних на батчі
p_batch_size = 32
training_loader = DataLoader(training_data, batch_size=p_batch_size, shuffle=True)
print(f"Датасет розбито на батчі з розміром {p_batch_size}")

# функції втрат
def discriminator_loss_fn(real_output, fake_output):
    return -(torch.mean(real_output) - torch.mean(fake_output))

def generator_loss_fn(fake_output):
    return -torch.mean(fake_output)

# gradient penalty для WGAN-GP
def gradient_penalty(i_discriminator, i_real_images, i_fake_images, i_conditions, i_device):
    batch_size = i_real_images.size(0)

    # Приведення зображень до векторного вигляду
    i_real_images = i_real_images.view(batch_size, -1)
    i_fake_images = i_fake_images.view(batch_size, -1)

    alpha = torch.rand(batch_size, 1).to(i_device)
    interpolated = (alpha * i_real_images + (1 - alpha) * i_fake_images).requires_grad_(True)

    # Конкатенація інтерпольованих зображень з умовами
    interpolated_input = torch.cat((interpolated, i_conditions), dim=1)

    # Пропуск через дискримінатор
    interpolated_output = i_discriminator(interpolated_input)

    # Обчислення градієнтів
    gradients = torch.autograd.grad(
        outputs=interpolated_output,
        inputs=interpolated,
        grad_outputs=torch.ones(interpolated_output.size()).to(i_device),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    # Обчислення штрафу за градієнт
    gradients = gradients.view(batch_size, -1)
    penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return penalty


# параметри моделей
p_latent_dim = 100
p_image_size = 28*28
p_condition_size = 10
p_negative_slope = 0.2
p_dropout = 0.2

mnist_generator = create_generator(p_latent_dim, p_condition_size, p_image_size)
mnist_discriminator = create_discriminator(p_image_size, p_condition_size, p_negative_slope, p_dropout)

def weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)

mnist_generator.apply(weights_init)
mnist_discriminator.apply(weights_init)

# параметри навчання
p_epochs = 20
p_lr_g = 0.0002
p_lr_d = 0.00005
p_beta1 = 0.0
p_beta2 = 0.9
p_discriminator_iterations = 5
lambda_gp = 10  # коефіцієнт для градієнтного штрафу

# оптимізатори
discriminator_optimizer = torch.optim.Adam(mnist_discriminator.parameters(), lr=p_lr_d, betas=(p_beta1, p_beta2))
generator_optimizer = torch.optim.Adam(mnist_generator.parameters(), lr=p_lr_g, betas=(p_beta1, p_beta2))

current_time = datetime.now().strftime("%H:%M:%S")
print(f"[{current_time}] Початок процесу навчання")

# процес навчання
mnist_discriminator.train()
mnist_generator.train()
for epoch in range(p_epochs):
    total_discriminator_loss = 0.0
    total_generator_loss = 0.0
    num_batches = len(training_loader)

    for i, (real_images, labels) in enumerate(training_loader):
        batch_size = real_images.size(0)
        real_images = real_images.view(batch_size, -1).to(device)
        real_conditions = torch.nn.functional.one_hot(labels, num_classes=p_condition_size).float().to(device)

        # === тренування дискримінатора ===
        for _ in range(p_discriminator_iterations):
            real_labels = torch.full((batch_size, 1), 0.9).to(device)
            fake_labels = torch.full((batch_size, 1), 0.1).to(device)

            real_input = torch.cat((real_images, real_conditions), dim=1)
            real_output_d = mnist_discriminator(real_input)


            # генерація фейкових даних
            noise = torch.randn(batch_size, p_latent_dim).to(device)
            fake_conditions = torch.eye(p_condition_size)[torch.randint(0, p_condition_size, (batch_size,))].to(device)
            generator_input = torch.cat((noise, fake_conditions), dim=1)
            fake_images = mnist_generator(generator_input)
            fake_input = torch.cat((fake_images, fake_conditions), dim=1)
            fake_output_d = mnist_discriminator(fake_input)

            discriminator_loss = discriminator_loss_fn(real_output_d, fake_output_d)
            gp = gradient_penalty(mnist_discriminator, real_images, fake_images, real_conditions, device)
            discriminator_loss += lambda_gp * gp

            discriminator_optimizer.zero_grad()
            discriminator_loss.backward()
            discriminator_optimizer.step()

            total_discriminator_loss += discriminator_loss.item()

        # === тренування генератора ===
        noise = torch.randn(batch_size, p_latent_dim).to(device)
        fake_conditions = torch.eye(p_condition_size)[torch.randint(0, p_condition_size, (batch_size,))].to(device)
        generator_input = torch.cat((noise, fake_conditions), dim=1)
        fake_images = mnist_generator(generator_input)

        fake_input = torch.cat((fake_images, fake_conditions), dim=1)
        fake_output_g = mnist_discriminator(fake_input)

        generator_loss = generator_loss_fn(fake_output_g)
        generator_optimizer.zero_grad()
        generator_loss.backward()
        generator_optimizer.step()

        total_generator_loss += generator_loss.item()

    # логування втрат
    avg_discriminator_loss = total_discriminator_loss / (num_batches * p_discriminator_iterations)
    avg_generator_loss = total_generator_loss / num_batches

    current_time = datetime.now().strftime("%H:%M:%S")
    print(f"[{current_time}] | Epoch [{epoch + 1}/{p_epochs}] | Discriminator Loss: {avg_discriminator_loss:.4f}, Generator Loss: {avg_generator_loss:.4f}")

    # приклад згенерованих зображень
    #if (epoch + 1) % 100 == 0:
    #    mnist_generator.eval()
    #    plot(generator=mnist_generator, images_num=16, latent_dim=p_latent_dim)
    #    mnist_generator.train()

    #if avg_generator_loss/avg_discriminator_loss > 5 or avg_discriminator_loss/avg_generator_loss > 5 :
    #    print("Рання зупинка.")
    #    break

torch.save(mnist_generator.state_dict(), "results/mnist_generator.pth")
print("Генератор збережено у файл 'mnist_generator.pth'")
