# =====================================================
# КЛАССИФИКАЦИЯ РАСТЕНИЙ - ПОЛНЫЙ ДАТАСЕТ (48 видов)
# =====================================================

import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
from sklearn.model_selection import train_test_split
import random
from collections import defaultdict

# ---------- 1. Путь к датасету ----------
dataset_path = "/kaggle/input/48-plant-leaves-datasets/Leaves of 48 Plants Dataset"
print(f"📁 Путь к датасету: {dataset_path}")

# ---------- 2. Получаем ВСЕ классы (папки) ----------
plant_classes = [d for d in os.listdir(dataset_path) 
                 if os.path.isdir(os.path.join(dataset_path, d))]
plant_classes.sort()
print(f"🌿 Найдено классов растений: {len(plant_classes)}")
print(f"📋 Первые 10: {plant_classes[:10]}")

# ---------- 3. Собираем все изображения по классам ----------
class_to_images = defaultdict(list)

for plant_class in plant_classes:
    class_path = os.path.join(dataset_path, plant_class)
    for img_file in os.listdir(class_path):
        if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
            class_to_images[plant_class].append(os.path.join(class_path, img_file))

# Статистика
print(f"\n📊 Статистика по классам:")
class_sizes = {cls: len(imgs) for cls, imgs in class_to_images.items()}
print(f"  Всего изображений: {sum(class_sizes.values())}")
print(f"  Среднее на класс: {np.mean(list(class_sizes.values())):.0f}")
print(f"  Максимум: {max(class_sizes.values())}")
print(f"  Минимум: {min(class_sizes.values())}")

# ---------- 4. Создаем структуру для YOLO ----------
temp_dir = '/content/plant_dataset_48'
os.makedirs(f'{temp_dir}/train', exist_ok=True)
os.makedirs(f'{temp_dir}/val', exist_ok=True)

print("\n📂 Создаем структуру для 48 классов...")

for plant_class, images in class_to_images.items():
    if len(images) < 2:
        print(f"  ⚠️ Пропускаем {plant_class}: только {len(images)} изображений")
        continue
    
    # Создаем папки для класса
    os.makedirs(f'{temp_dir}/train/{plant_class}', exist_ok=True)
    os.makedirs(f'{temp_dir}/val/{plant_class}', exist_ok=True)
    
    # Разделяем на train (80%) и val (20%)
    train_imgs, val_imgs = train_test_split(images, test_size=0.2, random_state=42)
    
    # Копируем файлы
    for img_path in train_imgs:
        shutil.copy2(img_path, f'{temp_dir}/train/{plant_class}/{os.path.basename(img_path)}')
    for img_path in val_imgs:
        shutil.copy2(img_path, f'{temp_dir}/val/{plant_class}/{os.path.basename(img_path)}')

# ---------- 5. Проверка ----------
train_classes = len(os.listdir(f'{temp_dir}/train'))
val_classes = len(os.listdir(f'{temp_dir}/val'))
train_total = sum(len(os.listdir(f'{temp_dir}/train/{cls}')) for cls in os.listdir(f'{temp_dir}/train'))
val_total = sum(len(os.listdir(f'{temp_dir}/val/{cls}')) for cls in os.listdir(f'{temp_dir}/val'))

print(f"\n✅ Итоговая структура:")
print(f"  Train: {train_total} изображений, {train_classes} классов")
print(f"  Val: {val_total} изображений, {val_classes} классов")

# ---------- 6. Обучение модели ----------
print("\n🏗️ Начинаем обучение на 48 классах...")

import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"🖥️ Устройство: {device.upper()}")

model = YOLO('yolov8n-cls.pt')

results = model.train(
    data=temp_dir,
    epochs=50,                     # Больше эпох для 48 классов
    imgsz=224,
    batch=32 if device == 'cuda' else 16,
    device=device,
    workers=4,
    patience=15,                   # Больше терпения
    project='plant_classification',
    name='48_plants_full',
    exist_ok=True,
    pretrained=True,
    verbose=True
)

# ---------- 7. Сохранение результатов ----------
# Список всех классов
with open('/content/plant_classes_48.txt', 'w') as f:
    for cls in plant_classes:
        f.write(f"{cls}\n")

print("\n" + "="*60)
print("🎉 ОБУЧЕНИЕ НА 48 КЛАССАХ ЗАВЕРШЕНО!")
print("="*60)
print(f"\n📁 Данные: {temp_dir}")
print(f"🤖 Модель: /content/runs/classify/plant_classification/48_plants_full/weights/best.pt")
print(f"📋 Список 48 классов: plant_classes_48.txt")
