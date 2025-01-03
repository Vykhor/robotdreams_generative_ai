from torch import nn

def create_discriminator(negative_slope):
    return nn.Sequential(
        nn.Conv2d(1, 128, kernel_size=4, stride=2, padding=1),
        nn.LeakyReLU(negative_slope=negative_slope),
        nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
        nn.BatchNorm2d(256),
        nn.LeakyReLU(negative_slope=negative_slope),
        nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
        nn.BatchNorm2d(512),
        nn.LeakyReLU(negative_slope=negative_slope),
        nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=0),
        nn.Sigmoid()
    )