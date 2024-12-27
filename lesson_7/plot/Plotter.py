import torch
import matplotlib.pyplot as plt
from torchvision.utils import make_grid

def plot(generator, images_num, latent_dim):
    # генерація випадкових векторів
    noise = torch.randn(images_num, latent_dim)

    print(f"Згенеровані вектори унікальні: {len(noise.unique(dim=0)) == images_num}")
    with torch.no_grad():
        generated_images = generator(noise).view(-1, 1, 28, 28)
        print(f"Кількість унікальних зображень: {len(generated_images.unique(dim=0))}")

    print(f"Мінімум: {generated_images.min().item()}")
    print(f"Максимум: {generated_images.max().item()}")
    print(f"Середнє: {generated_images.mean().item()}")

    # візуалізація
    grid = make_grid(generated_images, nrow=4, normalize=True)
    plt.imshow(grid.permute(1, 2, 0))
    plt.title("Generated Images")
    plt.axis("off")
    plt.show()