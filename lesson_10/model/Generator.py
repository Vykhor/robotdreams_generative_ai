from torch import nn

def create_generator(negative_slope):
    return nn.Sequential(
        # Вхід: пошкоджене зображення 32x32
        nn.Conv2d(in_channels=3, out_channels=64, kernel_size=4, stride=2, padding=1),  # -> 16x16
        nn.LeakyReLU(negative_slope, inplace=True),

        nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1),  # -> 8x8
        nn.BatchNorm2d(128),
        nn.LeakyReLU(negative_slope, inplace=True),

        nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1),  # -> 4x4
        nn.BatchNorm2d(256),
        nn.LeakyReLU(negative_slope, inplace=True),

        # Початок реконструкції
        nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1),  # -> 8x8
        nn.BatchNorm2d(128),
        nn.ReLU(inplace=True),

        nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1),  # -> 16x16
        nn.BatchNorm2d(64),
        nn.ReLU(inplace=True),

        nn.ConvTranspose2d(in_channels=64, out_channels=3, kernel_size=4, stride=2, padding=1),  # -> 32x32
        nn.Tanh()
    )
