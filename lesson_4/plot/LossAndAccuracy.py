import matplotlib.pyplot as plt

# Побудова графіків
def plot_loss_and_accuracy(train_losses, val_losses, accuracies):
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(20, 5))

    # Графік втрат на тренувальних даних
    plt.subplot(1, 3, 1)
    plt.plot(epochs, train_losses, label="Training Loss", color="red")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.legend()

    # Графік втрат на валідаційних даних
    plt.subplot(1, 3, 2)
    plt.plot(epochs, val_losses, label="Validation Loss", color="blue")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Validation Loss")
    plt.legend()

    # Графік точності
    plt.subplot(1, 3, 3)
    plt.plot(epochs, accuracies, label="Validation Accuracy", color="green")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Validation Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.show()
