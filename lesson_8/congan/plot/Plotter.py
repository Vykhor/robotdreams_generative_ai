import torch
import matplotlib.pyplot as plt
from torchvision.utils import make_grid

def plot(generator, images_num, latent_dim, condition_size, label):
    # генерація випадкових векторів
    noise = torch.randn(images_num, latent_dim)

    specified_conditions = torch.tensor([label] * images_num)
    one_hot_conditions = torch.nn.functional.one_hot(specified_conditions, num_classes=condition_size).float()

    generator_input = torch.cat((noise, one_hot_conditions), dim=1)
    with torch.no_grad():
        generated_images = generator(generator_input).view(-1, 1, 28, 28)
    # візуалізація
    grid = make_grid(generated_images, nrow=4, normalize=True)
    plt.imshow(grid.permute(1, 2, 0))
    plt.title("Generated Images")
    plt.axis("off")
    plt.show()