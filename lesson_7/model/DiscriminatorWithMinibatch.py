import torch
from torch import nn

class MinibatchDiscrimination(nn.Module):
    def __init__(self, in_features, out_features, kernel_dim):
        super(MinibatchDiscrimination, self).__init__()
        self.T = nn.Parameter(torch.randn(in_features, out_features * kernel_dim))
        self.out_features = out_features
        self.kernel_dim = kernel_dim

    def forward(self, x):
        batch_size = x.size(0)
        m = x @ self.T  # Розмір: (batch_size, out_features * kernel_dim)
        m = m.view(batch_size, self.out_features, self.kernel_dim)
        out = torch.zeros(batch_size, self.out_features, device=x.device)

        for i in range(batch_size):
            diff = m[i].unsqueeze(0) - m  # Різниця між поточним і всіма іншими
            abs_diff = torch.sum(torch.abs(diff), dim=2)  # Розмір: (batch_size, out_features)
            out[i] = torch.sum(torch.exp(-abs_diff), dim=0)  # Експонента для взаємодії

        return torch.cat([x, out], dim=1)  # Конкатенація оригінальних ознак

def create_discriminator(image_size, negative_slope,  dropout):
    return nn.Sequential(
        nn.Linear(image_size, 1024),
        nn.LeakyReLU(negative_slope=negative_slope),
        nn.Dropout(p=dropout),
        nn.Linear(1024, 512),
        nn.LeakyReLU(negative_slope=negative_slope),
        nn.Dropout(p=dropout),
        nn.Linear(512, 256),
        nn.LeakyReLU(negative_slope=negative_slope),
        nn.Dropout(p=dropout),
        MinibatchDiscrimination(256, out_features=100, kernel_dim=5),
        nn.Linear(256 + 100, 1),
        nn.Sigmoid()
)

def discriminator_with_features(discriminator, x):
    features = None

    def hook(_, __, i_output):
        nonlocal features
        features = i_output

    handle = discriminator[7].register_forward_hook(hook)
    output = discriminator(x)
    handle.remove()
    return output, features