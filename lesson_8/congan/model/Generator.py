from torch import nn

def create_generator(latent_dim, condition_size, image_size):
    input_size = latent_dim + condition_size
    return nn.Sequential(
    nn.Linear(input_size, 256),
    nn.ReLU(),
    nn.Linear(256, 512),
    nn.ReLU(),
    nn.Linear(512, 1024),
    nn.ReLU(),
    nn.Linear(1024, image_size),
    nn.Tanh()
)