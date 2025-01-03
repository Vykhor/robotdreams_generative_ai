from torch import nn

def create_generator(latent_dim):
    return nn.Sequential(
        nn.ConvTranspose2d(latent_dim, 256, kernel_size=4, stride=1, padding=0),  # Output: 256 x 4 x 4
        nn.BatchNorm2d(256),
        nn.ReLU(),
        nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # Output: 128 x 8 x 8
        nn.BatchNorm2d(128),
        nn.ReLU(),
        nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # Output: 64 x 16 x 16
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # Output: 32 x 32 x 32
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.ConvTranspose2d(32, 1, kernel_size=3, stride=1, padding=3),  # Output: 1 x 28 x 28
        nn.Tanh()
    )
