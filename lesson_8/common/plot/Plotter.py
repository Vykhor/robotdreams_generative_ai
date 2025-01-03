import torch
import matplotlib.pyplot as plt
from torchvision.utils import make_grid

def plot(generator, images_num, latent_dim):
    # генерація випадкових векторів
    noise = torch.randn(images_num, latent_dim, 1, 1)
    generated_images = generator(noise)

    # візуалізація
    grid = make_grid(generated_images, nrow=4, normalize=True)
    plt.imshow(grid.permute(1, 2, 0))
    plt.title("Generated Images")
    plt.axis("off")
    plt.show()