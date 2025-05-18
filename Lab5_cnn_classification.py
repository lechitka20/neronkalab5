# Импорт необходимых библиотек
import torch
import torch.nn as nn  # Модуль для создания нейронных сетей
import torch.optim as optim  # Оптимизаторы для обучения
from torchvision import datasets, models, transforms  # Работа с изображениями и предобученными моделями
from torch.utils.data import DataLoader, random_split  # Загрузка данных и разделение на выборки
from PIL import Image  # Для работы с изображениями
import os  # Для работы с файловой системой
import matplotlib.pyplot as plt  # Визуализация результатов
import numpy as np  # Числовые операции
import seaborn as sns  # Визуализация матрицы ошибок

# Установка переменной окружения для предотвращения конфликтов OpenMP
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

def check_and_skip_invalid_images(folder_path):
    """
    Проверяет и пропускает неподходящие изображения.
    Возвращает список путей к валидным изображениям.
    
    Args:
        folder_path (str): Путь к папке с изображениями
        
    Returns:
        list: Список путей к валидным изображениям
    """
    valid_images = []
    # Рекурсивно обходим все файлы в папке и подпапках
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                # Пытаемся открыть и проверить изображение
                img = Image.open(file_path)
                img.verify()  # Проверка целостности файла
                img.close()
                valid_images.append(file_path)
            except Exception as e:
                print(f"Skipping invalid image: {file_path} ({e})")
    return valid_images

# Путь к папке с данными 
data_path = './SportsEquipment'

# Проверяем и пропускаем неподходящие изображения
print("Checking and skipping invalid images...")
valid_image_paths = check_and_skip_invalid_images(data_path)

# Преобразования для данных:
# 1. Изменение размера до 224x224 (требование ResNet)
# 2. Преобразование в тензор PyTorch
# 3. Нормализация по средним и стандартным отклонениям ImageNet
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Стандартный размер для ResNet
    transforms.ToTensor(),  # Преобразование в тензор [0,1]
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Нормализация для ImageNet
                         std=[0.229, 0.224, 0.225])
])

class CustomImageDataset(torch.utils.data.Dataset):
    """
    Пользовательский класс датасета для загрузки изображений.
    Наследуется от torch.utils.data.Dataset.
    """
    def __init__(self, image_paths, transform=None):
        """
        Инициализация датасета.
        
        Args:
            image_paths (list): Список путей к изображениям
            transform (callable, optional): Преобразования для изображений
        """
        self.image_paths = image_paths
        self.transform = transform
        # Получаем список уникальных классов из структуры папок
        self.classes = sorted(set([os.path.basename(os.path.dirname(path)) 
                              for path in image_paths]))
        # Создаем словарь для преобразования имени класса в индекс
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
    
    def __len__(self):
        """Возвращает общее количество изображений в датасете."""
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        """
        Загружает и возвращает одно изображение и его метку по индексу.
        
        Args:
            idx (int): Индекс изображения
            
        Returns:
            tuple: (image, label) - изображение и его метка класса
        """
        image_path = self.image_paths[idx]
        try:
            # Открываем изображение и преобразуем в RGB (на случай если оно в градациях серого)
            image = Image.open(image_path).convert("RGB")
            # Получаем имя класса из структуры папок
            class_name = os.path.basename(os.path.dirname(image_path))
            # Преобразуем имя класса в числовой индекс
            label = self.class_to_idx[class_name]
            
            # Применяем преобразования если они заданы
            if self.transform:
                image = self.transform(image)
                
            return image, label
        except Exception as e:
            print(f"Error loading image at index {idx}: {e}")
            # В случае ошибки пропускаем файл и берем следующий
            return self.__getitem__((idx + 1) % len(self.image_paths))

# Создаем датасет из валидных изображений
dataset = CustomImageDataset(valid_image_paths, transform=transform)

# Разделяем данные на обучающую и тестовую выборки (80%/20%)
train_size = int(0.8 * len(dataset))  # 80% для обучения
test_size = len(dataset) - train_size  # Остальное для теста
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# Создаем DataLoader для удобной загрузки батчами
# Параметры:
# - batch_size: количество изображений в одном батче
# - shuffle: перемешивать ли данные (для обучающей выборки - да)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Выводим информацию о классах
print("Classes in dataset:", dataset.classes)

# Загружаем предобученную модель ResNet18 с весами, обученными на ImageNet
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

# Модифицируем последний полносвязный слой под наше количество классов
# ResNet18 изначально имеет 1000 выходов (по числу классов в ImageNet)
# Нам нужно заменить последний слой на слой с num_classes выходами
num_classes = len(dataset.classes)
model.fc = nn.Linear(model.fc.in_features, num_classes)

# Определяем устройство для обучения (GPU если доступно, иначе CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Перемещаем модель на выбранное устройство
model.to(device)

# Определяем функцию потерь (кросс-энтропия для многоклассовой классификации)
criterion = nn.CrossEntropyLoss()

# Выбираем оптимизатор (Adam с learning rate = 0.001)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Обучение модели
num_epochs = 10  # Количество эпох обучения
losses = []  # Для сохранения значений потерь

print("Starting training...")
for epoch in range(num_epochs):
    model.train()  # Переводим модель в режим обучения
    running_loss = 0.0  # Для накопления потерь за эпоху
    
    # Итерация по батчам обучающих данных
    for inputs, labels in train_loader:
        # Перемещаем данные на то же устройство, что и модель
        inputs, labels = inputs.to(device), labels.to(device)
        
        # Обнуляем градиенты с предыдущей итерации
        optimizer.zero_grad()
        
        # Прямой проход (forward pass)
        outputs = model(inputs)
        
        # Вычисляем функцию потерь
        loss = criterion(outputs, labels)
        
        # Обратный проход (backward pass) - вычисляем градиенты
        loss.backward()
        
        # Шаг оптимизации - обновляем веса
        optimizer.step()
        
        # Суммируем потери для статистики
        running_loss += loss.item()
    
    # Вычисляем средние потери за эпоху
    avg_loss = running_loss / len(train_loader)
    losses.append(avg_loss)
    
    # Выводим информацию о прогрессе
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

# Визуализация процесса обучения
plt.figure(figsize=(10, 5))
plt.plot(range(1, num_epochs + 1), losses, marker='o')
plt.title("Training Loss over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.show()

# Оценка модели на тестовых данных
model.eval()  # Переводим модель в режим оценки
correct = 0  # Счетчик правильных предсказаний
total = 0    # Общее количество примеров

# Отключаем вычисление градиентов для тестовой выборки
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        # Прямой проход
        outputs = model(inputs)
        
        # Получаем предсказанные классы (индекс с максимальной вероятностью)
        _, predicted = torch.max(outputs.data, 1)
        
        # Обновляем счетчики
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

# Вычисляем точность
accuracy = 100 * correct / total
print(f"Accuracy on test data: {accuracy:.2f}%")

# Создание матрицы ошибок для детального анализа
model.eval()
all_preds = []  # Для сохранения всех предсказаний
all_labels = [] # Для сохранения всех истинных меток

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)  # Получаем предсказанные классы
        
        # Сохраняем предсказания и истинные метки (перемещаем на CPU)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Создаем матрицу ошибок вручную
confusion_matrix = np.zeros((num_classes, num_classes), dtype=int)

# Заполняем матрицу ошибок
for true_label, pred_label in zip(all_labels, all_preds):
    confusion_matrix[true_label][pred_label] += 1

# Визуализация матрицы ошибок
plt.figure(figsize=(10, 8))
sns.heatmap(confusion_matrix, annot=True, fmt="d", 
            xticklabels=dataset.classes, 
            yticklabels=dataset.classes, 
            cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()