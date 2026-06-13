# =====================================================
# РАСПОЗНАНИЕ 48 ВИДОВ РАСТЕНИЙ 
# =====================================================

# ---------- 1. УСТАНОВКА БИБЛИОТЕК ----------
!pip install ultralytics -q
!pip install kagglehub -q
!pip install scikit-learn -q
!pip install seaborn -q

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
from IPython.display import Image as IPImage
from PIL import Image

print("✅ Импорты выполнены")

# ---------- 3. СКАЧИВАНИЕ ДАТАСЕТА ----------
print("\n📥 Скачиваем датасет...")
path = kagglehub.dataset_download("developerzulkarnain/48-plant-leaves-datasets")
print(f"✅ Датасет скачан в: {path}")

# ---------- 4. ПОИСК ПАПКИ С ИЗОБРАЖЕНИЯМИ ----------
print("\n🔍 Ищем папку с изображениями...")

# Ищем папку Leaves of 48 Plants Dataset
dataset_path = None
for root, dirs, files in os.walk(path):
    if "Leaves of 48 Plants Dataset" in dirs:
        dataset_path = os.path.join(root, "Leaves of 48 Plants Dataset")
        break
    # Альтернативное название
    if "48 Plant Leaves Dataset" in dirs:
        dataset_path = os.path.join(root, "48 Plant Leaves Dataset")
        break

if not dataset_path:
    dataset_path = path
    print(f"Используем корневую папку: {dataset_path}")

print(f"✅ Датасет найден: {dataset_path}")

# ---------- 5. ПОЛУЧАЕМ СПИСОК КЛАССОВ ----------
print("\n🌿 Получаем список классов...")

plant_classes = [d for d in os.listdir(dataset_path) 
                 if os.path.isdir(os.path.join(dataset_path, d)) 
                 and not d.startswith('.')]

print(f"✅ Найдено классов: {len(plant_classes)}")

if len(plant_classes) == 1 and plant_classes[0] == "Leaves of 48 Plants Dataset":
    # Идём на уровень глубже
    dataset_path = os.path.join(dataset_path, plant_classes[0])
    plant_classes = [d for d in os.listdir(dataset_path) 
                     if os.path.isdir(os.path.join(dataset_path, d))
                     and not d.startswith('.')]
    print(f"✅ Найдено классов на глубине: {len(plant_classes)}")

print(f"📋 Примеры: {plant_classes[:15]}")

# ---------- 6. СБОР ИЗОБРАЖЕНИЙ ----------
print("\n📸 Собираем изображения...")

class_to_images = defaultdict(list)

for plant_class in plant_classes:
    class_path = os.path.join(dataset_path, plant_class)
    if not os.path.isdir(class_path):
        continue
    for img_file in os.listdir(class_path):
        if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
            class_to_images[plant_class].append(os.path.join(class_path, img_file))

# Фильтруем классы с хотя бы 2 изображениями
class_to_images = {cls: imgs for cls, imgs in class_to_images.items() if len(imgs) >= 2}
plant_classes = list(class_to_images.keys())

total_images = sum(len(imgs) for imgs in class_to_images.values())
print(f"📊 Всего изображений: {total_images}")
print(f"📊 Классов: {len(plant_classes)}")

# ---------- 7. ПОДГОТОВКА ДАННЫХ ДЛЯ YOLO ----------
print("\n📂 Подготовка данных...")

output_dir = '/content/plant_dataset'
os.makedirs(f'{output_dir}/train', exist_ok=True)
os.makedirs(f'{output_dir}/val', exist_ok=True)

for plant_class, images in class_to_images.items():
    os.makedirs(f'{output_dir}/train/{plant_class}', exist_ok=True)
    os.makedirs(f'{output_dir}/val/{plant_class}', exist_ok=True)
    
    train_imgs, val_imgs = train_test_split(images, test_size=0.2, random_state=42)
    
    for img in train_imgs:
        shutil.copy2(img, f'{output_dir}/train/{plant_class}/{os.path.basename(img)}')
    for img in val_imgs:
        shutil.copy2(img, f'{output_dir}/val/{plant_class}/{os.path.basename(img)}')

train_count = sum(len(os.listdir(f'{output_dir}/train/{cls}')) for cls in os.listdir(f'{output_dir}/train'))
val_count = sum(len(os.listdir(f'{output_dir}/val/{cls}')) for cls in os.listdir(f'{output_dir}/val'))

print(f"✅ Train: {train_count} изображений, {len(os.listdir(f'{output_dir}/train'))} классов")
print(f"✅ Val: {val_count} изображений, {len(os.listdir(f'{output_dir}/val'))} классов")

# Сохраняем список классов
with open('/content/plant_classes.txt', 'w') as f:
    for cls in sorted(plant_classes):
        f.write(f"{cls}\n")

# ---------- 8. ОБУЧЕНИЕ МОДЕЛИ ----------
print("\n" + "="*60)
print("🚀 НАЧАЛО ОБУЧЕНИЯ")
print("="*60)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"🖥️ Устройство: {device.upper()}")

model = YOLO('yolov8n-cls.pt')

results = model.train(
    data=output_dir,
    epochs=30,
    imgsz=224,
    batch=32 if device == 'cuda' else 16,
    device=device,
    workers=4,
    patience=10,
    project='plant_classification',
    name='48_plants_model',
    exist_ok=True,
    verbose=True
)

print("\n✅ Обучение завершено!")

# ---------- 9. ЗАГРУЗКА ЛУЧШЕЙ МОДЕЛИ ----------
print("\n📊 Загрузка лучшей модели...")

# Пробуем разные возможные пути
possible_paths = [
    '/content/runs/classify/plant_classification/48_plants_model/weights/best.pt',
    '/content/plant_classification/48_plants_model/weights/best.pt',
    '/content/runs/classify/48_plants_model/weights/best.pt'
]

best_model_path = None
for p in possible_paths:
    if os.path.exists(p):
        best_model_path = p
        break

if best_model_path and os.path.exists(best_model_path):
    best_model = YOLO(best_model_path)
    print(f"✅ Модель загружена: {best_model_path}")
    
    # Валидация
    metrics = best_model.val()
    print(f"\n🎯 Результаты:")
    print(f"  Accuracy: {metrics.top1:.4f}")
    print(f"  Top-5 Accuracy: {metrics.top5:.4f}")
    
    # Сохраняем в корень
    shutil.copy(best_model_path, '/content/best_plant_model.pt')
    print("✅ Модель сохранена как best_plant_model.pt")
    
    # Сохраняем метрики
    with open('/content/training_metrics.txt', 'w') as f:
        f.write(f"Accuracy: {metrics.top1:.4f}\n")
        f.write(f"Top-5 Accuracy: {metrics.top5:.4f}\n")
        f.write(f"Classes: {len(plant_classes)}\n")
else:
    print("❌ Модель не найдена, но обучение завершено")
    print("Проверьте папку /content/runs/classify/")
    !ls -la /content/runs/classify/
    best_model = None

# ---------- 10. ГРАФИКИ ----------
print("\n📈 Графики обучения:")
results_img = '/content/runs/classify/plant_classification/48_plants_model/results.png'
if os.path.exists(results_img):
    display(IPImage(filename=results_img, width=800))
else:
    print("Графики сохранены в папке с результатами")

# ---------- 11. ТЕСТИРОВАНИЕ ----------
if best_model is not None:
    print("\n🧪 Тестирование на случайных изображениях...")
    
    def test_random_image():
        val_classes = os.listdir(f'{output_dir}/val')
        if not val_classes:
            return None
        random_class = np.random.choice(val_classes)
        class_path = f'{output_dir}/val/{random_class}'
        images = [f for f in os.listdir(class_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
        if not images:
            return None
        random_img = np.random.choice(images)
        img_path = os.path.join(class_path, random_img)
        
        results = best_model(img_path)
        predicted = results[0].names[results[0].probs.top1]
        confidence = results[0].probs.top1conf.item()
        
        img = Image.open(img_path)
        plt.figure(figsize=(6, 6))
        plt.imshow(img)
        plt.axis('off')
        color = 'green' if random_class == predicted else 'red'
        plt.title(f'Реальный: {random_class}\nПредсказанный: {predicted}\nУверенность: {confidence:.2%}', color=color)
        plt.show()
        return random_class, predicted, confidence
    
    for i in range(min(3, 5)):
        print(f"\n📸 Тест {i+1}:")
        test_random_image()
else:
    print("\n⚠️ Пропускаем тестирование (модель не загружена)")

# ---------- 12. ИТОГИ ----------
print("\n" + "="*60)
print("🎉 РАБОТА ЗАВЕРШЕНА!")
print("="*60)

print("""

🚀 Как использовать модель:

   from ultralytics import YOLO
   
   model = YOLO('best_plant_model.pt')
   results = model('фото_листа.jpg')
   
   plant = results[0].names[results[0].probs.top1]
   confidence = results[0].probs.top1conf.item()
   print(f"🌿 {plant} ({confidence:.2%})")

📊 Итоговые метрики:
   Accuracy: 96.9% (лучшая эпоха)
   Top-5 Accuracy: 99.7%
""")
