import torch
from torch.utils.data import DataLoader
from torch import nn, optim
from torchvision.utils import make_grid
import matplotlib.pyplot as plt

from lesson_7.data.Dataset import load_dataset_csv
from lesson_7.model.Discriminator import create_discriminator
from lesson_7.model.Generator import create_generator

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu")
print(f"Використовується пристрій: {device}")

training_data = load_dataset_csv("training_data.csv").to(device)
test_data = load_dataset_csv("test_data.csv").to(device)

# розбиття даних на батчі
p_batch_size = 32

training_loader = DataLoader(training_data, batch_size=p_batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=p_batch_size, shuffle=False)
print(f"Датасет розбито на батчі з розміром {p_batch_size}")

# параметри моделей
p_latent_dim = 100
p_image_size = 28*28
p_negative_slope = 0.2
p_dropout = 0.3

mnist_generator = create_generator(p_latent_dim, p_image_size)
mnist_discriminator = create_discriminator(p_image_size, p_negative_slope, p_dropout)

# параметри навчання
p_epochs = 100
p_lr = 0.0002
p_beta1 = 0.5
loss_fn = nn.BCELoss()

# оптимізатори
generator_optimizer = optim.Adam(mnist_generator.parameters(), lr=p_lr, betas=(p_beta1, 0.999))
discriminator_optimizer = optim.Adam(mnist_discriminator.parameters(), lr=p_lr, betas=(p_beta1, 0.999))

# процес навчання
for epoch in range(p_epochs):
    for real_images, _ in training_loader:
        batch_size = real_images.size(0)
        real_images = real_images.view(batch_size, -1).to(device)

        # === тренування дискримінатора ===
        real_labels = torch.ones(batch_size, 1).to(device)
        fake_labels = torch.zeros(batch_size, 1).to(device)

        real_output = mnist_discriminator(real_images)
        real_loss = loss_fn(real_output, real_labels)

        # генерація фейкових даних
        noise = torch.randn(batch_size, p_latent_dim).to(device)
        fake_images = mnist_generator(noise)

        fake_output = mnist_discriminator(fake_images.detach())
        fake_loss = loss_fn(fake_output, fake_labels)

        discriminator_loss = real_loss + fake_loss

        discriminator_optimizer.zero_grad()
        discriminator_loss.backward()
        discriminator_optimizer.step()

        # === Тренування генератора ===
        fake_output = mnist_discriminator(fake_images)
        generator_loss = loss_fn(fake_output, real_labels)  # Генератор хоче, щоб дискримінатор думав, що дані справжні

        generator_optimizer.zero_grad()
        generator_loss.backward()
        generator_optimizer.step()

torch.save(mnist_generator.state_dict(), "mnist_generator.pth")
print("Генератор збережено у файл 'mnist_generator.pth'")

