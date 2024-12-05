import torch
import matplotlib.pyplot as plt

def plot_predictions(model, test_loader):
    model.eval()
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            # Відображення кількох зображень
            for i in range(6):  # Відображаємо перші 6 зображень
                plt.subplot(2, 3, i + 1)  # Сітка 2x3
                plt.imshow(images[i][0].numpy(), cmap='gray')  # Зображення
                plt.title(f"True: {labels[i]}, Pred: {predicted[i]}")
                plt.axis('off')
            plt.show()
            break