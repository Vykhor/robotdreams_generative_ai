import pandas as pd
import torch
from datetime import datetime
from torch.utils.data import DataLoader, TensorDataset
from torch import nn, optim

from lesson_7.model.Discriminator import create_discriminator, discriminator_with_features
from lesson_7.model.Generator import create_generator
import torch.nn.functional as f

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

training_data = load_dataset_csv("./data/training_data.csv")
print(f"Розмір навчального набору: {len(training_data)}")

# розбиття даних на батчі
p_batch_size = 16

training_loader = DataLoader(training_data, batch_size=p_batch_size, shuffle=True)

print(f"Датасет розбито на батчі з розміром {p_batch_size}")

# параметри моделей
p_latent_dim = 100
p_image_size = 28*28
p_negative_slope = 0.2
p_dropout = 0.3

mnist_generator = create_generator(p_latent_dim, p_image_size)
mnist_discriminator = create_discriminator(p_image_size, p_negative_slope, p_dropout)

# параметри навчання
p_epochs = 50
p_lr_g = 0.0002
p_lr_d = 0.000001
p_beta1 = 0.5
p_beta2 = 0.999
p_weight_decay = 0.00001
loss_fn = nn.BCELoss()

# оптимізатори
discriminator_optimizer = optim.Adam(mnist_discriminator.parameters(), lr=p_lr_d, betas=(p_beta1, p_beta2), weight_decay=p_weight_decay)
generator_optimizer = optim.Adam(mnist_generator.parameters(), lr=p_lr_g, betas=(p_beta1, p_beta2), weight_decay=p_weight_decay)

current_time = datetime.now().strftime("%H:%M:%S")
print(f"[{current_time}] Початок процесу навчання")

def feature_matching_loss(i_real_features, i_fake_features):
    return f.mse_loss(i_real_features, i_fake_features)

# процес навчання
mnist_discriminator.train()
mnist_generator.train()
for epoch in range(p_epochs):
    total_discriminator_loss = 0.0
    total_generator_loss = 0.0
    num_batches = len(training_loader)

    for i, (real_images, _) in enumerate(training_loader):
        batch_size = real_images.size(0)
        real_images = real_images.view(batch_size, -1).to(device)

        # === тренування дискримінатора ===
        real_labels = torch.ones(batch_size, 1).to(device)
        fake_labels = torch.zeros(batch_size, 1).to(device)

        discriminator_optimizer.zero_grad()

        real_output_d  = mnist_discriminator(real_images)
        real_loss = loss_fn(real_output_d, real_labels)

        # генерація фейкових даних
        noise = torch.randn(batch_size, p_latent_dim).to(device)
        fake_images = mnist_generator(noise)
        fake_output_d = mnist_discriminator(fake_images.detach())

        fake_loss = loss_fn(fake_output_d, fake_labels)

        discriminator_loss = real_loss + fake_loss

        discriminator_optimizer.zero_grad()
        discriminator_loss.backward()
        discriminator_optimizer.step()

        # === Тренування генератора ===
        for _ in range(2):
            noise = torch.randn(batch_size, p_latent_dim).to(device)
            fake_images = mnist_generator(noise)

            real_output_g, real_features_g = discriminator_with_features(mnist_discriminator, real_images.detach())
            fake_output_g, fake_features_g = discriminator_with_features(mnist_discriminator, fake_images)

            generator_loss = loss_fn(fake_output_g, real_labels)

            feature_loss = feature_matching_loss(i_real_features=real_features_g, i_fake_features=fake_features_g)

            generator_with_features_loss = generator_loss + 0.5 * feature_loss

            generator_optimizer.zero_grad()
            generator_with_features_loss.backward()
            generator_optimizer.step()

            total_discriminator_loss += discriminator_loss.item()
            total_generator_loss += generator_with_features_loss.item()

    # логування втрат
    avg_discriminator_loss = total_discriminator_loss / num_batches
    avg_generator_loss = total_generator_loss / (num_batches  * 2)

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
