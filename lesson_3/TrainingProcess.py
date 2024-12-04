import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import pandas as pd

from lesson_3.plot.ConfusionMatrix import plot_confusion_matrix
from lesson_3.plot.LossAndAccuracy import plot_loss_and_accuracy
from lesson_3.plot.Predictions import plot_predictions
from MnistModel import create_model
import EarlyStopping

device = torch.device(
    "cuda" if torch.cuda.is_available() 
    else "xpu" if torch.xpu.is_available()
    else "cpu")
print(f"Використовується пристрій: {device}")

# підготовка даних
# завантаження відфільтрованих даних
def load_dataset_csv(file_path):
    df = pd.read_csv(file_path)
    labels = torch.tensor(df['label'].values, dtype=torch.long)
    images = torch.tensor(df.drop(columns=['label']).values, dtype=torch.float32).view(-1, 1, 28, 28)
    print(f"Датасет завантажено з файлу: {file_path}")
    return TensorDataset(images, labels)

training_data = load_dataset_csv("training_data.csv")
test_data = load_dataset_csv("test_data.csv")

# розбиття даних на батчі
p_batch_size = 32

# створення валідаційної вибірки
train_size = int(0.8 * len(training_data))
val_size = len(training_data) - train_size
train_dataset, val_dataset = random_split(training_data, [train_size, val_size])

train_loader = DataLoader(training_data, batch_size=p_batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=p_batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=p_batch_size, shuffle=False)
print(f"Датасет розбито на батчі з розміром {p_batch_size}")

# параметри
p_input_size = training_data[0][0].numel()  # 784
p_output_classes_number = len(set(label.item() for _, label in training_data))  # 10
p_hidden_features = 512
p_learning_rate = 0.0005
p_num_epochs = 50
p_dropout = 0.3
p_weight_decay = 0.000001

# створення моделі
model = create_model(input_size=p_input_size, output_classes_number=p_output_classes_number, hidden_features=p_hidden_features, dropout=p_dropout).to(device)

# функція втрат та оптимізатор
criterion = nn.CrossEntropyLoss() # має вбудовану функцію Softmax
optimizer = optim.Adam(model.parameters(), lr=p_learning_rate, weight_decay=p_weight_decay) # оптимізатор з ваговою регуляризацією
print(f"Створено criterion CrossEntropyLoss та optimizer Adam з weight_decay = {p_weight_decay}")

# списки для збереження даних для побудови графіку втрат і точності під час навчання
train_losses = []
val_losses = []
val_accuracies = []

# ініціалізація EarlyStopping
early_stopping = EarlyStopping.EarlyStopping(patience=5, delta=0.001)

# навчання моделі
model.train()
for epoch in range(p_num_epochs):
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

    # оцінка на валідаційних даних
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    avg_val_loss = val_loss / len(val_loader)
    val_losses.append(avg_val_loss)
    val_accuracy = 100 * correct / total
    val_accuracies.append(val_accuracy)

    print(f"Epoch {epoch+1}/{p_num_epochs}, Train Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}%")

    # перевірка EarlyStopping
    early_stopping(avg_val_loss)
    if early_stopping.early_stop:
        print("Спрацювала перевірка EarlyStopping.")
        break

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
plot_loss_and_accuracy(train_losses=train_losses, val_losses=val_losses, accuracies=val_accuracies)
plot_predictions(model=model, test_loader=test_loader)
