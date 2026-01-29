#!/usr/bin/env python3
"""
Основной скрипт для запуска пайплайна обработки данных.
Использование: python app.py путь/к/hh.csv
"""

import argparse
import os
import sys
import gc
import numpy as np
from pathlib import Path
import psutil
import humanize

# Добавляем src в путь для импорта
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.pipeline.pipeline import DataProcessingPipeline
from src.utils.logger import setup_logger


def get_memory_usage():
    """Возвращает текущее использование памяти."""
    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss
    return humanize.naturalsize(mem)


def main():
    """Основная функция для запуска пайплайна."""
    parser = argparse.ArgumentParser(
        description="Пайплайн обработки данных HH с использованием цепочки ответственности"
    )
    parser.add_argument(
        "filepath",
        type=str,
        help="Путь к CSV файлу с данными"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Тестовый режим (обработать только 10000 строк)"
    )
    
    args = parser.parse_args()
    
    # Проверяем существование файла
    if not os.path.exists(args.filepath):
        print(f"Ошибка: файл '{args.filepath}' не найден!")
        sys.exit(1)
    
    # Настраиваем логирование
    logger = setup_logger()
    
    try:
        # Логируем начало
        logger.info("=" * 60)
        logger.info("ЗАПУСК ПАЙПЛАЙНА ОБРАБОТКИ HH ДАННЫХ")
        logger.info(f"Файл: {args.filepath}")
        logger.info(f"Использование памяти в начале: {get_memory_usage()}")
        logger.info("=" * 60)
        
        # Создаем и запускаем пайплайн
        pipeline = DataProcessingPipeline(logger)
        
        # В тестовом режиме создаем временный файл
        if args.test:
            logger.info("ТЕСТОВЫЙ РЕЖИМ: создаем тестовый файл")
            test_filepath = f"test_{os.path.basename(args.filepath)}"
            
            # Читаем первые 10000 строк
            import pandas as pd
            df_test = pd.read_csv(args.filepath, nrows=10000)
            df_test.to_csv(test_filepath, index=False)
            
            logger.info(f"Создан тестовый файл: {test_filepath} ({len(df_test)} строк)")
            file_to_process = test_filepath
        else:
            file_to_process = args.filepath
        
        # Обрабатываем данные
        logger.info(f"Начало обработки файла: {file_to_process}")
        X, y = pipeline.process(file_to_process)
        
        # Очищаем память
        gc.collect()
        logger.info(f"Использование памяти после обработки: {get_memory_usage()}")
        
        # Сохраняем результаты
        output_dir = Path(file_to_process).parent
        output_path_x = output_dir / "X_data.npy"
        output_path_y = output_dir / "y_data.npy"
        
        # Сохраняем с индикатором прогресса
        logger.info(f"Сохранение X_data.npy ({X.shape})...")
        X.save(output_path_x)
        
        logger.info(f"Сохранение y_data.npy ({y.shape})...")
        y.save(output_path_y)
        
        # Удаляем временный тестовый файл
        if args.test and os.path.exists(test_filepath):
            os.remove(test_filepath)
            logger.info(f"Удален тестовый файл: {test_filepath}")
        
        # Логируем успех
        logger.info("=" * 60)
        logger.info("ОБРАБОТКА УСПЕШНО ЗАВЕРШЕНА")
        logger.info(f"  Признаки (X): {output_path_x}")
        logger.info(f"    Размер: {X.shape}, Тип: {X.dtype}")
        logger.info(f"  Целевая переменная (y): {output_path_y}")
        logger.info(f"    Размер: {y.shape}, Тип: {y.dtype}")
        logger.info(f"Использование памяти в конце: {get_memory_usage()}")
        logger.info("=" * 60)
        
        # Вывод в консоль
        print(f"\n{'✅' * 30}")
        print(f"✅ ОБРАБОТКА ЗАВЕРШЕНА УСПЕШНО!")
        print(f"{'✅' * 30}")
        print(f"📁 X_data.npy: {output_path_x}")
        print(f"   Размер: {X.shape} | Тип: {X.dtype}")
        print(f"📁 y_data.npy: {output_path_y}")
        print(f"   Размер: {y.shape} | Тип: {y.dtype}")
        
        # Размер файлов
        if output_path_x.exists():
            size_x = output_path_x.stat().st_size / (1024**2)
            print(f"   Размер файла X: {size_x:.1f} MB")
        
        if output_path_y.exists():
            size_y = output_path_y.stat().st_size / (1024**2)
            print(f"   Размер файла y: {size_y:.1f} MB")
        
        print(f"\n💡 Совет: Проверьте пропуски в y_data.npy:")
        print(f"   np.isnan(y).sum() = {np.isnan(y).sum()}")
        
    except Exception as e:
        logger.error(f"Ошибка при выполнении пайплайна: {e}", exc_info=True)
        print(f"\n{'❌' * 30}")
        print(f"❌ ПРОИЗОШЛА ОШИБКА: {e}")
        print(f"{'❌' * 30}")
        sys.exit(1)


if __name__ == "__main__":
    main()