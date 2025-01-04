from torch import nn

def create_discriminator(image_size, condition_size, negative_slope,  dropout):
    input_size = image_size + condition_size
    return nn.Sequential(
        nn.Linear(input_size, 1024),
        nn.LeakyReLU(negative_slope=negative_slope),
        nn.Dropout(p=dropout),
        nn.Linear(1024, 512),
        nn.LeakyReLU(negative_slope=negative_slope),
        nn.Dropout(p=dropout),
        nn.Linear(512, 256),
        nn.LeakyReLU(negative_slope=negative_slope),
        nn.Dropout(p=dropout),
        nn.Linear(256, 1)
)