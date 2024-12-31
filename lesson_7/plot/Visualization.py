import torch

from lesson_7.model.Generator import create_generator
from lesson_7.plot.Plotter import plot

# створення генератора
latent_dim = 100
image_size = 28 * 28
mnist_generator = create_generator(latent_dim, image_size)

# відновлення стану генератора зі збереженого файлу
mnist_generator.load_state_dict(torch.load("../results/3/mnist_generator_3.pth"))
mnist_generator.eval()  # переведення моделі в режим оцінювання
print("Генератор завантажено")

plot(generator=mnist_generator, images_num=16, latent_dim=latent_dim)