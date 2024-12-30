from torch import nn

def create_generator(latent_dim, image_size, negative_slope):
    return nn.Sequential(
    nn.Linear(latent_dim, 256),
    nn.LeakyReLU(negative_slope=negative_slope),
    nn.Linear(256, 512),
    nn.LeakyReLU(negative_slope=negative_slope),
    nn.Linear(512, 1024),
    nn.LeakyReLU(negative_slope=negative_slope),
    nn.Linear(1024, image_size),
    nn.Tanh()
)