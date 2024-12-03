from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import torch
import matplotlib.pyplot as plt

def plot_confusion_matrix(model, test_loader, training_data_classes, device):
    # Збирання істинних та передбачених значень
    y_true = []  # Істинні мітки
    y_pred = []  # Передбачені мітки

    model.eval()
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            y_true.extend(labels.cpu().numpy())  # Додаємо істинні мітки
            y_pred.extend(predicted.cpu().numpy())  # Додаємо передбачені мітки

    # Побудова матриці невідповідностей
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=training_data_classes)

    # Відображення матриці
    disp.plot()
    plt.title("Confusion Matrix")
    plt.show()
