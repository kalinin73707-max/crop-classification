# =====================================================
# КЛАССИФИКАЦИЯ 48 ВИДОВ РАСТЕНИЙ ПО ЛИСТЬЯМ
# Полный готовый код для Google Colab
# =====================================================

# ---------- 1. УСТАНОВКА БИБЛИОТЕК ----------
!pip install ultralytics -q
!pip install kagglehub -q
!pip install scikit-learn -q
!pip install seaborn -q
!pip install opencv-python-headless -q

print("✅ Библиотеки установлены")

# ---------- 2. ИМПОРТЫ ----------
import kagglehub
import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
from sklearn.model_selection import train_test_split
from collections import defaultdict
import torch
import zipfile
from IPython.display import Image as IPImage

print("✅ Импорты выполнены")

# ---------- 3. СКАЧИВАНИЕ ДАТАСЕТА ----------
print("\n📥 Скачиваем датасет с 48 видами растений...")

try:
    # Пытаемся скачать через kagglehub
    path = kagglehub.dataset_download("developerzulkarnain/48-plant-leaves-datasets")
    print(f"✅ Датасет скачан в: {path}")
except Exception as e:
    print(f"Ошибка kagglehub: {e}")
    print("Пробуем альтернативный способ...")
    
    # Альтернативный способ: через Kaggle API
    !mkdir -p ~/.kaggle
    !echo '{"username":"your_username","key":"your_key"}' > ~/.kaggle/kaggle.json
    !chmod 600 ~/.kaggle/kaggle.json
    !kaggle datasets download -d developerzulkarnain/48-plant-leaves-datasets
    !unzip -q 48-plant-leaves-datasets.zip -d /content/dataset
    path = "/content/dataset"

# ---------- 4. ПОИСК ПАПКИ С ИЗОБРАЖЕНИЯМИ ----------
print("\n🔍 Ищем папку с изображениями...")

dataset_path = None

# Список возможных путей
possible_paths = [
    path,
    os.path.join(path, "Leaves of 48 Plants Dataset"),
    os.path.join(path, "48 Plant Leaves Dataset"),
    "/kaggle/input/48-plant-leaves-datasets/Leaves of 48 Plants Dataset",
    "/content/dataset/Leaves of 48 Plants Dataset",
]

for p in possible_paths:
    if os.path.exists(p):
        dataset_path = p
        print(f"✅ Найдено: {dataset_path}")
        break

# Если не нашли, ищем рекурсивно
if not dataset_path:
    for root, dirs, files in os.walk(path):
        for dir_name in dirs:
            if "leaf" in dir_name.lower() or "plant" in dir_name.lower():
                dataset_path = os.path.join(root, dir_name)
                print(f"✅ Найдено рекурсивно: {dataset_path}")
                break
        if dataset_path:
            break

if not dataset_path:
    print("❌ Папка с изображениями не найдена!")
    print("Содержимое корневой папки:")
    !ls -la {path}
    raise FileNotFoundError("Не удалось найти папку с изображениями")

# ---------- 5. ПОЛУЧАЕМ СПИСОК КЛАССОВ ----------
print("\n🌿 Получаем список классов растений...")

plant_classes = [d for d in os.listdir(dataset_path) 
                 if os.path.isdir(os.path.join(dataset_path, d)) 
                 and not d.startswith('.') 
                 and not d.startswith('_')]
plant_classes.sort()

print(f"✅ Найдено классов: {len(plant_classes)}")
print(f"📋 Примеры: {plant_classes[:15]}")

# Если классов слишком мало, пробуем другой уровень вложенности
if len(plant_classes) < 10:
    print("⚠️ Классов меньше ожидаемого. Ищем глубже...")
    for subdir in os.listdir(dataset_path):
        subdir_path = os.path.join(dataset_path, subdir)
        if os.path.isdir(subdir_path):
            subclasses = [d for d in os.listdir(subdir_path) 
                         if os.path.isdir(os.path.join(subdir_path, d))]
            if len(subclasses) > len(plant_classes):
                dataset_path = subdir_path
                plant_classes = subclasses
                print(f"✅ Найдено классов на новом уровне: {len(plant_classes)}")
                break

# ---------- 6. СБОР ИЗОБРАЖЕНИЙ ----------
print("\n📸 Собираем изображения...")

class_to_images = defaultdict(list)

for plant_class in plant_classes:
    class_path = os.path.join(dataset_path, plant_class)
    if not os.path.isdir(class_path):
        continue
    
    for img_file in os.listdir(class_path):
        if img_file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            img_path = os.path.join(class_path, img_file)
            class_to_images[plant_class].append(img_path)

# Удаляем классы без изображений
plant_classes = [cls for cls in plant_classes if class_to_images[cls]]
class_to_images = {cls: imgs for cls, imgs in class_to_images.items() if imgs}

# Статистика
total_images = sum(len(imgs) for imgs in class_to_images.values())
print(f"📊 Всего изображений: {total_images}")
print(f"📊 Классов с изображениями: {len(plant_classes)}")

if total_images == 0:
    print("❌ Нет изображений! Проверьте структуру датасета.")
    print("Структура папки:")
    !ls -la {dataset_path}
    for sub in os.listdir(dataset_path)[:5]:
        sub_path = os.path.join(dataset_path, sub)
        if os.path.isdir(sub_path):
            print(f"  {sub}/: {len(os.listdir(sub_path))} файлов")
    raise ValueError("Датасет не содержит изображений")

# ---------- 7. ПОДГОТОВКА ДАННЫХ ДЛЯ YOLO ----------
print("\n📂 Подготавливаем данные для YOLO...")

output_dir = '/content/plant_dataset'
os.makedirs(f'{output_dir}/train', exist_ok=True)
os.makedirs(f'{output_dir}/val', exist_ok=True)

# Проверяем минимальное количество изображений в классе
min_images = 2
filtered_classes = {}

for plant_class, images in class_to_images.items():
    if len(images) >= min_images:
        filtered_classes[plant_class] = images
    else:
        print(f"  ⚠️ Пропущен класс '{plant_class}': только {len(images)} изображений")

print(f"✅ Классов для обучения: {len(filtered_classes)}")

# Копируем файлы в структуру YOLO
for plant_class, images in filtered_classes.items():
    # Создаем папки для класса
    os.makedirs(f'{output_dir}/train/{plant_class}', exist_ok=True)
    os.makedirs(f'{output_dir}/val/{plant_class}', exist_ok=True)
    
    # Разделяем на train (80%) и val (20%)
    train_imgs, val_imgs = train_test_split(images, test_size=0.2, random_state=42)
    
    # Копируем
    for img_path in train_imgs:
        shutil.copy2(img_path, f'{output_dir}/train/{plant_class}/{os.path.basename(img_path)}')
    for img_path in val_imgs:
        shutil.copy2(img_path, f'{output_dir}/val/{plant_class}/{os.path.basename(img_path)}')

# Проверка результата
train_total = sum(len(os.listdir(f'{output_dir}/train/{cls}')) for cls in os.listdir(f'{output_dir}/train'))
val_total = sum(len(os.listdir(f'{output_dir}/val/{cls}')) for cls in os.listdir(f'{output_dir}/val'))

print(f"\n✅ Данные готовы:")
print(f"  Train: {train_total} изображений, {len(os.listdir(f'{output_dir}/train'))} классов")
print(f"  Val: {val_total} изображений, {len(os.listdir(f'{output_dir}/val'))} классов")

# ---------- 8. СОХРАНЕНИЕ СПИСКА КЛАССОВ ----------
with open('/content/plant_classes.txt', 'w') as f:
    for cls in sorted(filtered_classes.keys()):
        f.write(f"{cls}\n")
print("✅ Список классов сохранён в plant_classes.txt")

# ---------- 9. ОБУЧЕНИЕ МОДЕЛИ ----------
print("\n" + "="*60)
print("🚀 НАЧАЛО ОБУЧЕНИЯ МОДЕЛИ")
print("="*60)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"🖥️ Устройство: {device.upper()}")

if device == 'cuda':
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  Видеопамять: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# Параметры обучения
EPOCHS = 30  # Для быстрого теста, можно увеличить до 50-100
BATCH_SIZE = 32 if device == 'cuda' else 16
IMAGE_SIZE = 224

print(f"📊 Параметры:")
print(f"  Эпох: {EPOCHS}")
print(f"  Батч: {BATCH_SIZE}")
print(f"  Размер: {IMAGE_SIZE}")

# Загружаем модель
model = YOLO('yolov8n-cls.pt')
print("✅ Модель загружена")

# Обучение
results = model.train(
    data=output_dir,
    epochs=EPOCHS,
    imgsz=IMAGE_SIZE,
    batch=BATCH_SIZE,
    device=device,
    workers=4,
    patience=10,
    project='plant_classification',
    name='48_plants_model',
    exist_ok=True,
    pretrained=True,
    verbose=True,
    seed=42
)

print("\n✅ Обучение завершено!")

# ---------- 10. ВАЛИДАЦИЯ ----------
print("\n📊 Валидация модели...")

# Путь к лучшей модели
best_model_path = '/content/plant_classification/48_plants_model/weights/best.pt'

if os.path.exists(best_model_path):
    best_model = YOLO(best_model_path)
    metrics = best_model.val()
    
    print(f"\n🎯 Результаты:")
    print(f"  Accuracy: {metrics.top1:.4f}")
    print(f"  Top-5 Accuracy: {metrics.top5:.4f}")
    print(f"  Loss: {metrics.loss:.4f}")
    
    # Сохраняем метрики
    with open('/content/training_metrics.txt', 'w') as f:
        f.write(f"Accuracy: {metrics.top1:.4f}\n")
        f.write(f"Top-5 Accuracy: {metrics.top5:.4f}\n")
        f.write(f"Loss: {metrics.loss:.4f}\n")
        f.write(f"Classes: {len(filtered_classes)}\n")
        f.write(f"Epochs: {EPOCHS}\n")
    
    # Копируем модель в корень
    shutil.copy(best_model_path, '/content/best_plant_model.pt')
    print("✅ Модель сохранена как best_plant_model.pt")
else:
    print("❌ Модель не найдена!")

# ---------- 11. ВИЗУАЛИЗАЦИЯ ----------
print("\n📈 Графики обучения:")

results_img = '/content/plant_classification/48_plants_model/results.png'
if os.path.exists(results_img):
    display(IPImage(filename=results_img, width=800))
else:
    print("Графики будут доступны после завершения обучения")

# ---------- 12. ТЕСТИРОВАНИЕ ----------
print("\n🧪 Тестирование на случайных изображениях...")

def test_random_image():
    val_classes = os.listdir(f'{output_dir}/val')
    if not val_classes:
        print("Нет валидационных классов")
        return
    
    random_class = np.random.choice(val_classes)
    class_path = f'{output_dir}/val/{random_class}'
    images = [f for f in os.listdir(class_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    if not images:
        return
    
    random_img = np.random.choice(images)
    img_path = os.path.join(class_path, random_img)
    
    results = best_model(img_path)
    predicted = results[0].names[results[0].probs.top1]
    confidence = results[0].probs.top1conf.item()
    
    # Отображаем
    from PIL import Image
    img = Image.open(img_path)
    plt.figure(figsize=(6, 6))
    plt.imshow(img)
    plt.axis('off')
    color = 'green' if random_class == predicted else 'red'
    plt.title(f'Реальный: {random_class}\nПредсказанный: {predicted}\nУверенность: {confidence:.2%}', color=color)
    plt.show()
    
    return random_class, predicted, confidence

# Тестируем 3 изображения
for i in range(min(3, 10)):
    print(f"\nТест {i+1}:")
    test_random_image()

# ---------- 13. ИТОГИ ----------
print("\n" + "="*60)
print("🎉 РАБОТА ЗАВЕРШЕНА УСПЕШНО!")
print("="*60)

print("""
📁 Файлы для скачивания (панель слева → 📂 Файлы):
   🤖 best_plant_model.pt - обученная модель
   📋 plant_classes.txt - список классов
   📊 training_metrics.txt - метрики

🚀 Как использовать модель:
   from ultralytics import YOLO
   model = YOLO('best_plant_model.pt')
   results = model('фото_листа.jpg')
   print(results[0].names[results[0].probs.top1])
""")
