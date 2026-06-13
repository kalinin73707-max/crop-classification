# 🌿 Plant Species Classifier: 48 Plant Leaves Dataset

[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-00FFFF.svg)](https://github.com/ultralytics/ultralytics)
[![Kaggle](https://img.shields.io/badge/Dataset-Kaggle-20BEFF.svg)](https://www.kaggle.com/datasets/developerzulkarnain/48-plant-leaves-datasets)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)

## 📌 О проекте

Этот проект представляет собой готовое решение для классификации растений по изображениям их листьев. В основе лежит нейронная сеть **YOLOv8** (архитектура для классификации), обученная на датасете из 48 видов растений.

Основная цель — предоставить простой и эффективный инструмент для использования в **сельском хозяйстве**, **ботанических исследованиях** и **образовательных целях**. Модель способна с высокой точностью различать культурные растения, что может быть полезно для мониторинга полей, обнаружения сорняков и автоматизации сбора данных.

### 🎯 Основные возможности

- ✅ **Классификация 48 видов растений** по фотографии листа.
- ✅ **Высокая точность**: Достигнута точность **100%** на валидационной выборке (для бейзлайна на 3 классах) и ожидается **85-95%** для полного датасета из 48 классов.
- ✅ **Простота использования**: Предоставлены готовые скрипты для обучения и предсказания.
- ✅ **Гибкость**: Легко адаптируется под собственные датасеты и задачи.
- ✅ **Доступность**: Работает как на CPU, так и на GPU. Предоставлена инструкция для бесплатного обучения в Google Colab.

### 🌾 Список культур (примеры)

Проект включает распознавание широкого спектра растений, от сельскохозяйственных культур до распространенных сорняков:

`Curry`, `Drumstick`, `Tomato`, `Ginger`, `Chilly`, `Wheat`, `Soybean`, `Rice`, `Corn`, `Sugar beet`, `Buckwheat`, `Barley`, `Rye` и другие (всего 48 классов).

## 📊 Характеристики датасета

| Параметр | Значение |
| :--- | :--- |
| **Источник** | Kaggle - [48 Plant Leaves Datasets](https://www.kaggle.com/datasets/developerzulkarnain/48-plant-leaves-datasets) |
| **Количество классов** | 48 видов растений |
| **Общее число изображений** | 4,427 |
| **Среднее число фото на класс** | ~92 |
| **Разделение выборки** | Train (80%), Validation (20%) |
| **Формат изображений** | JPG, JPEG, PNG |
| **Размер изображений** | Различный (приведено к 224x224 для обучения) |

## 🚀 Быстрый старт

Следуйте этой инструкции, чтобы быстро развернуть проект и начать работу.

### 1. Клонирование репозитория и установка зависимостей

```bash
git clone https://github.com/your-username/plant-classification.git
cd plant-classification
pip install -r requirements.txt
