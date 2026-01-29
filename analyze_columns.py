#!/usr/bin/env python3
"""
Скрипт для анализа колонок в hh.csv
"""

import pandas as pd
import sys
from pathlib import Path

def analyze_columns(filepath, max_rows=10000):
    """Анализирует колонки файла."""
    print("🔍 Анализ структуры файла...")
    print(f"Файл: {filepath}")
    
    # Проверяем размер
    size_mb = Path(filepath).stat().st_size / (1024 * 1024)
    print(f"Размер: {size_mb:.1f} МБ")
    
    try:
        # Читаем только заголовки
        print("\n📋 Заголовки колонок:")
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            header = f.readline().strip()
        
        # Определяем разделитель
        for sep in [',', ';', '\t', '|']:
            if sep in header:
                columns = header.split(sep)
                print(f"Разделитель: '{sep}'")
                print(f"Всего колонок: {len(columns)}")
                break
        
        # Показываем все колонки
        for i, col in enumerate(columns, 1):
            print(f"{i:3d}. {col}")
        
        # Читаем небольшой образец для анализа типов
        print("\n📊 Анализ типов данных (по 10к строк):")
        df_sample = pd.read_csv(
            filepath,
            nrows=max_rows,
            sep=sep,
            low_memory=False,
            encoding='utf-8',
            on_bad_lines='skip'
        )
        
        # Статистика по колонкам
        print("\n📈 Статистика по колонкам:")
        for col in df_sample.columns[:15]:  # первые 15 колонок
            dtype = df_sample[col].dtype
            unique = df_sample[col].nunique()
            missing = df_sample[col].isnull().sum()
            missing_pct = (missing / len(df_sample)) * 100
            
            # Предполагаем, какие могут быть target columns
            is_numeric = pd.api.types.is_numeric_dtype(dtype)
            is_low_cardinality = unique < 100  # мало уникальных значений
            
            print(f"\n{col}:")
            print(f"  Тип: {dtype}")
            print(f"  Уникальных: {unique}")
            print(f"  Пропусков: {missing} ({missing_pct:.1f}%)")
            
            # Подсказки для целевой переменной
            if is_numeric:
                print(f"  🔹 Кандидат в target (числовая)")
                if unique > 10:
                    print(f"    Возможно: цена, зарплата, рейтинг")
            elif is_low_cardinality:
                print(f"  🔸 Кандидат в target (категориальная)")
                print(f"    Возможно: класс, категория, статус")
            
            if unique < 10:
                print(f"  Примеры: {df_sample[col].unique()[:5]}")
        
        # Автоматический подбор кандидатов
        print("\n🎯 РЕКОМЕНДУЕМЫЕ КОЛОНКИ ДЛЯ target_column:")
        candidates = []
        
        for col in df_sample.columns:
            dtype = df_sample[col].dtype
            unique = df_sample[col].nunique()
            missing = df_sample[col].isnull().sum()
            
            # Критерии для хорошего target
            if pd.api.types.is_numeric_dtype(dtype):
                if missing < len(df_sample) * 0.5:  # не более 50% пропусков
                    if 2 <= unique <= 1000:  # не слишком много уникальных
                        candidates.append((col, "numeric", unique))
            elif unique <= 10:  # категориальные с малым числом классов
                if missing < len(df_sample) * 0.3:
                    candidates.append((col, "categorical", unique))
        
        if candidates:
            for col, col_type, unique in sorted(candidates, key=lambda x: x[2])[:10]:
                print(f"  - {col} ({col_type}, {unique} уникальных)")
        else:
            print("  Не найдено очевидных кандидатов. Возможные варианты:")
            for col in df_sample.columns:
                if any(keyword in col.lower() for keyword in 
                      ['salary', 'price', 'cost', 'target', 'class', 
                       'score', 'rating', 'status', 'result']):
                    print(f"  - {col}")
        
        # Сохраняем список колонок в файл
        with open('columns_list.txt', 'w', encoding='utf-8') as f:
            f.write("\n".join(df_sample.columns))
        
        print(f"\n💾 Список всех колонок сохранен в 'columns_list.txt'")
        
        return df_sample.columns.tolist()
        
    except Exception as e:
        print(f"Ошибка: {e}")
        return None

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Использование: python analyze_columns.py путь/к/hh.csv")
        sys.exit(1)
    
    analyze_columns(sys.argv[1])