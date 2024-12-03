import matplotlib.pyplot as plt
from MnistModel import num_epochs
from MnistModel import train_losses
from MnistModel import  test_accuracies

# Побудова графіків
epochs = range(1, num_epochs + 1)

# Графік втрат
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(epochs, train_losses, label="Train Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.legend()

# Графік точності
plt.subplot(1, 2, 2)
plt.plot(epochs, test_accuracies, label="Test Accuracy", color="green")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.title("Test Accuracy")
plt.legend()

plt.tight_layout()
plt.show()
