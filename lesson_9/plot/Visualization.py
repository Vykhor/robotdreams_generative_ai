import torch
from matplotlib import pyplot as plt

from lesson_9.data.Corrupting import corrupt_images
from lesson_9.data.Dataloader import get_dataloader
from lesson_9.model.Generator import create_generator

# Створення генератора
negative_slope = 0.2
mnist_generator = create_generator(negative_slope=negative_slope)

# Відновлення стану генератора зі збереженого файлу
mnist_generator.load_state_dict(torch.load("../result/generator.pth"))
mnist_generator.eval()  # Переведення моделі в режим оцінювання
print("Генератор завантажено")

# Завантаження тестових даних
test_file_path = "../cifar10_test_data.csv"
batch_size = 10
test_loader = get_dataloader(test_file_path, batch_size=batch_size, shuffle=True)

# Отримання прикладів відновлення
for batch in test_loader:
    real_images = batch[0]  # Оригінальні зображення
    with torch.no_grad():
        damaged_images = corrupt_images(real_images)  # Пошкоджуємо зображення

    with torch.no_grad():
        restored_images = mnist_generator(damaged_images)  # Відновлюємо зображення

    # Візуалізація сітки з прикладами
    fig, axes = plt.subplots(
        nrows=3,
        ncols=batch_size,
        figsize=(15, 6),
        gridspec_kw={'wspace': 0.0, 'hspace': 0.0},  # Прибираємо проміжки між зображеннями
    )

    # Перший рядок: оригінальні зображення
    for col in range(batch_size):
        axes[0, col].imshow(real_images[col].permute(1, 2, 0).cpu().numpy().clip(0, 1))
        axes[0, col].axis('off')

    # Другий рядок: пошкоджені зображення
    for col in range(batch_size):
        axes[1, col].imshow(damaged_images[col].permute(1, 2, 0).cpu().numpy().clip(0, 1))
        axes[1, col].axis('off')

    # Третій рядок: відновлені зображення
    for col in range(batch_size):
        axes[2, col].imshow(restored_images[col].permute(1, 2, 0).cpu().numpy().clip(0, 1))
        axes[2, col].axis('off')

    # Відображення
    plt.tight_layout(pad=0)  # Прибираємо зовнішні відступи
    plt.show()

    # Відображаємо лише один батч для демонстрації
    break
