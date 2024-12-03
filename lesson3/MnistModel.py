import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Використовується пристрій: {device}")

# завантаження даних, перетворення у тензори, нормалізація
training_data = datasets.MNIST(
    root='./datasets',
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.MNIST(
    root='./datasets',
    train=False,
    download=True,
    transform=ToTensor()
)

# перевірка та фільтрація датасету
def filter_missing_values(data):
    valid_indices = []
    for i, (image, label) in enumerate(data):
        if not torch.isnan(image).any() and label is not None:
            valid_indices.append(i)
        else:
            print(f"Пропущені значення в зображенні {i}, видалено.")
    return torch.utils.data.Subset(data, valid_indices)

def filter_anomalies(data):
    valid_indices = []
    for i, (image, label) in enumerate(data):
        if 0 <= image.min() <= image.max() <= 1 and image.size() == (1, 28, 28):
            valid_indices.append(i)
        else:
            print(f"Аномалія у зображенні {i}: розмір {image.size()} або значення пікселів, видалено.")
    return torch.utils.data.Subset(data, valid_indices)

def filter_data_types(data):
    valid_indices = []
    for i, (image, label) in enumerate(data):
        if isinstance(image, torch.Tensor) and isinstance(label, int):
            valid_indices.append(i)
        else:
            print(f"Некоректні типи даних у зображенні {i}: тип {type(image)} або мітки {type(label)}, видалено.")
    return torch.utils.data.Subset(data, valid_indices)

def subset_to_dataset(subset):
    images = []
    labels = []
    for i in range(len(subset)):
        image, label = subset[i]
        images.append(image)
        labels.append(label)
    images_tensor = torch.stack(images)  # Розмір: [N, 1, 28, 28]
    labels_tensor = torch.tensor(labels)  # Розмір: [N]
    return TensorDataset(images_tensor, labels_tensor)

print("Перевірка training_data")
training_data = filter_missing_values(training_data)
training_data = filter_anomalies(training_data)
training_data = filter_data_types(training_data)
training_data = subset_to_dataset(training_data)

print("Перевірка test_data")
test_data = filter_missing_values(test_data)
test_data = filter_anomalies(test_data)
test_data = filter_data_types(test_data)
test_data = subset_to_dataset(test_data)

print(f"Кількість зображень у training_data після перевірки: {len(training_data)}")
print(f"Кількість зображень у test_data після перевірки: {len(test_data)}")



# параметри
input_size = training_data[0][0].numel()  # 784
output_classes_number = len(set(label.item() for _, label in training_data))  # 10
hidden_features = 392
batch_size = 32
learning_rate = 0.2
num_epochs = 5

# створення класу моделі
class MnistModel(nn.Module):
  def __init__(self, input_size, output_classes_number):
    super(MnistModel, self).__init__()
    self.input = nn.Linear(input_size, hidden_features)
    self.relu1 = nn.ReLU()
    self.hidden = nn.Linear(hidden_features, hidden_features)
    self.relu2 = nn.ReLU()
    self.output = nn.Linear(hidden_features, output_classes_number)
    self.softmax = nn.Softmax(dim=1)

  def forward(self, x):
    x = x.view(x.size(0), -1)
    x = self.input(x)
    x = self.relu1(x)
    x = self.hidden(x)
    x = self.relu2(x)
    x = self.output(x)
    x = self.softmax(x)
    return x

# створення моделі
model = MnistModel(input_size, output_classes_number).to(device)
print(model)

# розбиття даних на батчі
train_loader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

# функція втрат та оптимізатор
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

# списки для збереження даних для побудови графіку втрат і точності під час навчання
train_losses = []
test_accuracies = []

# навчання моделі
model.train()
for epoch in range(num_epochs):
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    avg_train_loss = running_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    test_accuracies.append(accuracy)

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_train_loss:.4f}, Accuracy: {accuracy:.2f}%")
    model.train()

# оцінка моделі
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Accuracy: {100 * correct / total:.2f}%")
