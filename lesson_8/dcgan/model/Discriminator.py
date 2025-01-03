from torch import nn

def create_discriminator(negative_slope):
    return nn.Sequential(
        nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1),  # Output: 64 x 14 x 14
        nn.LeakyReLU(negative_slope=negative_slope),
        nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # Output: 128 x 7 x 7
        nn.BatchNorm2d(128),
        nn.LeakyReLU(negative_slope=negative_slope),
        nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=2),  # Output: 256 x 4 x 4
        nn.BatchNorm2d(256),
        nn.LeakyReLU(negative_slope=negative_slope),
        nn.Conv2d(256, 1, kernel_size=4, stride=1, padding=0),  # Output: 1 x 1 x 1
        nn.Sigmoid()
    )