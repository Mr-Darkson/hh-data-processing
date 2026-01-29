#!/usr/bin/env python3
"""
Скрипт для проверки результатов обработки.
"""

import numpy as np
import sys

def check_results():
    """Проверяет созданные файлы."""
    try:
        # Загружаем данные
        X = np.load('X_data.npy')
        y = np.load('y_data.npy')
        
        print("=" * 60)
        print("ПРОВЕРКА РЕЗУЛЬТАТОВ ОБРАБОТКИ")
        print("=" * 60)
        
        print(f"\n📊 Статистика по X_data.npy:")
        print(f"  Размер: {X.shape}")
        print(f"  Тип данных: {X.dtype}")
        print(f"  Минимальное значение: {np.nanmin(X):.4f}")
        print(f"  Максимальное значение: {np.nanmax(X):.4f}")
        print(f"  Среднее значение: {np.nanmean(X):.4f}")
        print(f"  Пропущенные значения: {np.isnan(X).sum()}")
        
        print(f"\n📊 Статистика по y_data.npy:")
        print(f"  Размер: {y.shape}")
        print(f"  Тип данных: {y.dtype}")
        print(f"  Минимальная зарплата: {np.nanmin(y):,.0f} руб.")
        print(f"  Максимальная зарплата: {np.nanmax(y):,.0f} руб.")
        print(f"  Средняя зарплата: {np.nanmean(y):,.0f} руб.")
        print(f"  Медианная зарплата: {np.nanmedian(y):,.0f} руб.")
        print(f"  Пропущенные значения: {np.isnan(y).sum()} ({np.isnan(y).sum()/len(y)*100:.1f}%)")
        
        print(f"\n📈 Распределение зарплат:")
        percentiles = [10, 25, 50, 75, 90, 95]
        for p in percentiles:
            value = np.nanpercentile(y, p)
            print(f"  {p}% перцентиль: {value:,.0f} руб.")
        
        print(f"\n✅ Все файлы успешно созданы и загружены!")
        
        # Пример использования для ML
        print(f"\n💡 Пример использования для машинного обучения:")
        print(f"  from sklearn.model_selection import train_test_split")
        print(f"  X_train, X_test, y_train, y_test = train_test_split(")
        print(f"      X, y, test_size=0.2, random_state=42)")
        print(f"  print(f'Обучающая выборка: {X_train.shape}')")
        print(f"  print(f'Тестовая выборка: {X_test.shape}')")
        
    except FileNotFoundError as e:
        print(f"❌ Файл не найден: {e}")
        print("Запустите сначала: python app.py hh.csv")
    except Exception as e:
        print(f"❌ Ошибка: {e}")

if __name__ == "__main__":
    check_results()