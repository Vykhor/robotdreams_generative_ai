import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
import pandas as pd

from ConfusionMatrix import plot_confusion_matrix
from LossAndAccuracy import plot_loss_and_accuracy

device = torch.device("xpu" if torch.xpu.is_available() else "cpu")
print(f"Використовується пристрій: {device}")

# завантаження відфільтрованих даних
def load_dataset_csv(file_path):
    df = pd.read_csv(file_path)
    labels = torch.tensor(df['label'].values, dtype=torch.long)
    images = torch.tensor(df.drop(columns=['label']).values, dtype=torch.float32).view(-1, 1, 28, 28)
    print(f"Датасет завантажено з файлу: {file_path}")
    return TensorDataset(images, labels)

training_data = load_dataset_csv("training_data.csv")
test_data = load_dataset_csv("test_data.csv")

# параметри
input_size = training_data[0][0].numel()  # 784
output_classes_number = len(set(label.item() for _, label in training_data))  # 10
hidden_features = 128
batch_size = 128
learning_rate = 0.001
num_epochs = 3
dropout = 0.3
weight_decay = 0.0001

# створення класу моделі
class MnistModel(nn.Module):
  def __init__(self, input_size, output_classes_number):
    super(MnistModel, self).__init__()
    self.input = nn.Linear(input_size, hidden_features)
    self.relu = nn.ReLU()
    self.hidden1 = nn.Linear(hidden_features, hidden_features)
    self.hidden2 = nn.Linear(hidden_features, hidden_features)
    #self.hidden3 = nn.Linear(hidden_features, hidden_features)
    self.output = nn.Linear(hidden_features, output_classes_number)
    self.dropout = nn.Dropout(dropout)

  def forward(self, x):
    x = x.view(x.size(0), -1)
    x = self.input(x)
    x = self.relu(x)
    x = self.hidden1(x)
    x = self.dropout(x)
    x = self.relu(x)
    x = self.hidden2(x)
    #x = self.dropout(x)
    #x = self.relu(x)
    #x = self.hidden3(x)
    x = self.output(x)
    return x

# створення моделі
model = MnistModel(input_size, output_classes_number).to(device)
print(model)

# розбиття даних на батчі
train_loader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

# функція втрат та оптимізатор
criterion = nn.CrossEntropyLoss() # має вбудовану функцію Softmax
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay) # оптимізатор з ваговою регуляризацією

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

plot_confusion_matrix(model=model, test_loader=test_loader, device=device, training_data_classes=set(label.item() for _, label in training_data))
plot_loss_and_accuracy(num_epochs=num_epochs, train_losses=train_losses, test_accuracies=test_accuracies)
