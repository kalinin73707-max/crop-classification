#!/usr/bin/env python3
"""
Plant Species Classification - Prediction Script
Распознавание растений на новых изображениях
"""

from ultralytics import YOLO
import sys
import argparse

def predict(image_path, model_path='best_plant_model.pt'):
    """
    Распознает растение на изображении
    
    Args:
        image_path: путь к изображению
        model_path: путь к обученной модели
    """
    # Загружаем модель
    model = YOLO(model_path)
    
    # Предсказание
    results = model(image_path)
    
    # Получаем результат
    predicted_class = results[0].names[results[0].probs.top1]
    confidence = results[0].probs.top1conf.item()
    
    print(f"\n🌿 Растение: {predicted_class}")
    print(f"📊 Уверенность: {confidence:.2%}")
    
    # Показываем топ-3
    print(f"\n🏆 Топ-3:")
    for i, (idx, conf) in enumerate(zip(results[0].probs.top3, results[0].probs.top3conf), 1):
        plant_name = results[0].probs.names[int(idx)]
        print(f"  {i}. {plant_name}: {conf:.2%}")
    
    return predicted_class, confidence

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Распознавание растений')
    parser.add_argument('image', help='Путь к изображению')
    parser.add_argument('--model', default='best_plant_model.pt', help='Путь к модели')
    
    args = parser.parse_args()
    predict(args.image, args.model)
