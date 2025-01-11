import torch
from matplotlib import pyplot as plt

from lesson_9.data.Corrupting import corrupt_images
from lesson_9.data.Dataloader import get_dataloader
from lesson_9.model.Generator import create_generator

# створення генератора
negative_slope = 0.2
mnist_generator = create_generator(negative_slope=negative_slope)

# відновлення стану генератора зі збереженого файлу
mnist_generator.load_state_dict(torch.load("../generator.pth"))
mnist_generator.eval()  # переведення моделі в режим оцінювання
print("Генератор завантажено")

# Завантаження тестових даних
test_file_path = "../cifar10_test_data.csv"
batch_size = 4
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
        nrows=batch_size + 1,
        ncols=3,
        figsize=(10, 3 * (batch_size + 1)),
        gridspec_kw={'wspace': 0.05, 'hspace': 0.05},  # Зменшення проміжків
    )

    # Додаємо заголовки над першим рядком
    titles = ["Оригінальне", "Пошкоджене", "Відновлене"]
    for col, title in enumerate(titles):
        axes[0, col].set_title(title, fontsize=16)
        axes[0, col].axis('off')  # Прибираємо осі для заголовків

    for i in range(batch_size):
        # Оригінальне зображення
        axes[i + 1, 0].imshow(real_images[i].permute(1, 2, 0).cpu().numpy().clip(0, 1))
        axes[i + 1, 0].axis('off')

        # Пошкоджене зображення
        axes[i + 1, 1].imshow(damaged_images[i].permute(1, 2, 0).cpu().numpy().clip(0, 1))
        axes[i + 1, 1].axis('off')

        # Відновлене зображення
        axes[i + 1, 2].imshow(restored_images[i].permute(1, 2, 0).cpu().numpy().clip(0, 1))
        axes[i + 1, 2].axis('off')

    # Налаштування зовнішніх меж
    plt.tight_layout(rect=[0, 0, 1, 0.98])  # Коригування зовнішніх меж для зручного вигляду
    plt.show()

    # Відображаємо лише один батч для демонстрації
    break
