import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import pandas as pd
import itertools

from MnistModel import create_model

device = torch.device("xpu" if torch.xpu.is_available() else "cpu")
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

# Розділення тренувальних даних на тренувальний та валідаційний набори (80% на тренування, 20% на валідацію)
train_size = int(0.8 * len(training_data))
val_size = len(training_data) - train_size
train_dataset, val_dataset = random_split(training_data, [train_size, val_size])
print(f"Датасет розбито на тренувальний ({train_size}) та валідаційний ({val_size})")

# розбиття даних на батчі
p_batch_size = 32

train_loader = DataLoader(train_dataset, batch_size=p_batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=p_batch_size, shuffle=True)
print(f"Датасет розбито на батчі з розміром {p_batch_size}")

# Можливі значення гіперпараметрів
p_hidden_features_options = [128, 256, 512]
p_learning_rate_options = [0.001, 0.0005, 0.0001, 0.00005]
p_dropout_options = [0.2, 0.3, 0.4, 0.5]
p_weight_decay_options = [0.00001, 0.000005, 0.000001]

hyperparameter_combinations = list(itertools.product(p_hidden_features_options, p_learning_rate_options, p_dropout_options, p_weight_decay_options))

# параметри
p_input_size = training_data[0][0].numel()  # 784
p_output_classes_number = len(set(label.item() for _, label in training_data))  # 10
p_num_epochs = 5

best_hyperparameters = None
best_accuracy = 0

# перебір гіперпараметрів
for p_hidden_features, p_learning_rate, p_dropout, p_weight_decay in hyperparameter_combinations:
    # створення моделі
    model = create_model(input_size=p_input_size, output_classes_number=p_output_classes_number, hidden_features=p_hidden_features, dropout=p_dropout).to(device)

    # функція втрат та оптимізатор
    criterion = nn.CrossEntropyLoss() # має вбудовану функцію Softmax
    optimizer = optim.Adam(model.parameters(), lr=p_learning_rate, weight_decay=p_weight_decay) # оптимізатор з ваговою регуляризацією
    print(f"Створено criterion CrossEntropyLoss та optimizer Adam з learning_rate={p_learning_rate} та weight_decay = {p_weight_decay}")

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

    # оцінка моделі на валідаційних даних
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_accuracy = 100 * correct / total
    print(f"Accuracy: {val_accuracy:.2f}%")
    print(f"------------------------------")

    if val_accuracy > best_accuracy:
        best_accuracy = val_accuracy
        best_hyperparameters = (p_hidden_features, p_learning_rate, p_dropout, p_weight_decay)

print(f"Найкраща комбінація гіперпараметрів: p_hidden_features={best_hyperparameters[0]}, p_learning_rate={best_hyperparameters[1]}, p_dropout={best_hyperparameters[2]}, p_weight_decay={best_hyperparameters[3]}")
print(f"Найкраща точність: {best_accuracy:.2f}%")
