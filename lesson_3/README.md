# Проєкт Класифікації MNIST

Цей репозиторій містить набір Python-скриптів для побудови, навчання та оцінки моделі глибокого навчання для класифікації рукописних цифр із датасету MNIST. Проєкт демонструє ключові компоненти машинного навчання, зокрема попередню обробку даних, налаштування гіперпараметрів, тренування, оцінку та візуалізацію.

---

## Особливості

- **Кастомна нейронна мережа**: Реалізація повнозв'язної моделі глибокого навчання на основі PyTorch.
- **Обробка даних**:
  - Попередня обробка датасету MNIST.
  - Розподіл даних на тренувальний, валідаційний та тестовий набори.
- **Процес тренування**: Включає моніторинг втрат і точності.
- **Налаштування гіперпараметрів**: Підтримка перебору комбінацій гіперпараметрів.
- **Візуалізація**:
  - Побудова графіків втрат і точності.
  - Генерація матриці невідповідностей для аналізу результатів класифікації.

---

## Опис скриптів

### 1. **`DatasetFiltration.py`**
Реалізує завантаження, валідацію та фільтрацію даних MNIST:
- Завантажує dataset MNIST.
- Перевіряє завантажений dataset на наявність пропущених значень, аномалій та некоректних типів даних.
- Відфільтровує некоректні дані.
- Зберігає відфільтровані тренувальні та тестові дані у файли для подальшого використання у навчанні моделей.

### 2. **`MnistModel.py`**
Містить опис архітектури нейронної мережі:
- Повнозв'язні шари з активацією ReLU.
- Налаштовуваний Dropout для регуляризації.
- Функція `create_model` для ініціалізації моделі з гнучкими параметрами.

### 3. **`EvaluateParams.py`**
Реалізує налаштування гіперпараметрів:
- Здійснює перебір значень для швидкості навчання, Dropout, кількості нейронів у прихованих шарах і вагової регуляризації.
- Оцінює модель на валідаційному наборі, щоб визначити оптимальні параметри.

### 4. **`plot/ConfusionMatrix.py`**
Генерує матрицю невідповідностей для оцінки результатів класифікації:
- Використовує `sklearn.metrics` для обчислення та візуалізації.
- Відображає матрицю за допомогою matplotlib.

### 5. **`plot/LossAndAccuracy.py`**
Побудова графіків втрат і точності:
- Два графіки: втрати за епохами та точність за епохами.
- Допомагає візуалізувати процес навчання.

### 6. **`plot/Prediction.py`**
Візуалізація передбачень:
- Візуалізує передбачення моделі на прикладі декількох зображень.

### 7. **`TrainingProcess.py`**
Реалізує основний процес тренування:
- Завантаження MNIST, розділення на тренувальний і тестовий набори.
- Конфігурація моделі, оптимізатора та функції втрат.
- Візуалізація результатів (графіки, матриця невідповідностей, передбачення).

---

## Результати

У папці `results` знаходять результати роботи `EvaluateParams.py` та візуалізації побудовані для моделі, навченої з цими параметрами.

---

## Використання
- Запустіть `DatasetFiltration.py` для завантаження трнувальних та тестових даних.
- Запустіть `EvaluateParams.py` для визначення оптимальних параметрів моделі.
- Пропишіть визначені параметри в `TrainingProcess.py`.
- Запустіть `TrainingProcess.py`.

---

### Попередні вимоги
- Python 3.8 або новіший.
- Бібліотеки PyTorch та Torchvision.
- Додаткові пакети: `matplotlib`, `pandas`, `sklearn`.