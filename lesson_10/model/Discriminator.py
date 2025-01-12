from torch import nn

def create_discriminator(negative_slope):
    return nn.Sequential(
        # Вхід: зображення 32x32
        nn.Conv2d(in_channels=3, out_channels=64, kernel_size=4, stride=2, padding=1),  # -> 16x16
        nn.LeakyReLU(negative_slope, inplace=True),
        nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1),  # -> 8x8
        nn.BatchNorm2d(128),
        nn.LeakyReLU(negative_slope, inplace=True),
        nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1),  # -> 4x4
        nn.BatchNorm2d(256),
        nn.LeakyReLU(negative_slope, inplace=True),
        # Останній згортковий шар: розмір зменшується до 1x1
        nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=1, padding=0),  # -> 1x1
        nn.LeakyReLU(negative_slope, inplace=True),
        # Вихідний шар: одне значення на виході
        nn.Conv2d(in_channels=512, out_channels=1, kernel_size=1),  # -> 1x1
        nn.Flatten(),
        nn.Sigmoid()
    )
