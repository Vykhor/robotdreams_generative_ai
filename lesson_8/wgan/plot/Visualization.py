import torch
from lesson_8.wgan.model.Generator import create_generator
from lesson_8.wgan.plot.Plotter import plot

# створення генератора
latent_dim = 100
image_size = 28*28
mnist_generator = create_generator(latent_dim=latent_dim, image_size=image_size)

# відновлення стану генератора зі збереженого файлу
mnist_generator.load_state_dict(torch.load("../results/mnist_generator.pth"))
mnist_generator.eval()  # переведення моделі в режим оцінювання
print("Генератор завантажено")

plot(generator=mnist_generator, images_num=16, latent_dim=latent_dim)