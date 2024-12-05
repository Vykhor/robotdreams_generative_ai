import torch.nn as nn


class MnistCNNModel(nn.Module):
    def __init__(self, output_classes_number, dropout):
        super(MnistCNNModel, self).__init__()
        # згортковий шар 1
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)  # Batch Normalization після Conv2d

        # згортковий шар 2
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        # згортковий шар 3
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        # активація
        self.relu = nn.ReLU()
        # MaxPooling для зменшення розмірності
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # Dropout для регуляризації
        self.dropout = nn.Dropout(dropout)

        # повнозв'зні шари
        self.fc1 = nn.Linear(128 * 7 * 7, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, output_classes_number)

    def forward(self, x):
        # Шар 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        # Шар 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.pool(x)  # MaxPooling

        # Шар 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.pool(x)  # MaxPooling

        # перетворення вектора для повнозв’язного шару (Flatten)
        x = x.view(x.size(0), -1)

        # повнозв’язні шари (Dense)
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.fc3(x)

        return x


def create_model(output_classes_number, dropout):
    model = MnistCNNModel(output_classes_number=output_classes_number, dropout=dropout)
    print(f"Created MnistCNNModel model: {model}")
    return model
