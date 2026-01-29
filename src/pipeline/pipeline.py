"""
Основной класс пайплайна обработки данных.
"""

import yaml
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
import numpy as np

from .base_handler import Handler
from .handlers import (
    ChunkedDataLoaderHandler,
    DataCleanerHandler,
    HHDataParserHandler,
    FeatureEngineeringHandler,
    DataSplitterHandler,
    ScalerHandler,
    FinalizerHandler
)


class DataProcessingPipeline:
    """
    Основной класс пайплайна обработки данных с цепочкой ответственности.
    """
    
    def __init__(self, logger=None):
        """
        Инициализирует пайплайн обработки данных.
        
        Args:
            logger: Логгер для записи событий
        """
        self.logger = logger
        self.config = self._load_config()
        
    def _load_config(self) -> Dict[str, Any]:
        """
        Загружает конфигурацию из YAML файла.
        
        Returns:
            Словарь с конфигурацией
        """
        config_path = Path(__file__).parent.parent.parent / "config.yaml"
        
        if not config_path.exists():
            if self.logger:
                self.logger.warning(f"Конфигурационный файл не найден: {config_path}")
            return {}
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        if self.logger:
            self.logger.info(f"Загружена конфигурация: target_column={config.get('data', {}).get('target_column')}")
        
        return config
    
    def _build_chain(self) -> Handler:
        """
        Строит цепочку обработчиков для HH данных.
        
        Returns:
            Первый обработчик в цепочке
        """
        # Создаем все обработчики в правильном порядке
        loader = ChunkedDataLoaderHandler(self.config)
        cleaner = DataCleanerHandler(self.config)
        hh_parser = HHDataParserHandler(self.config)
        feature_eng = FeatureEngineeringHandler(self.config)
        splitter = DataSplitterHandler(self.config)
        scaler = ScalerHandler(self.config)
        finalizer = FinalizerHandler(self.config)
        
        # Строим цепочку
        loader.set_next(cleaner).set_next(hh_parser).set_next(
            feature_eng
        ).set_next(splitter).set_next(scaler).set_next(finalizer)
        
        if self.logger:
            self.logger.info("Цепочка обработчиков построена")
        
        return loader
    
    def process(self, filepath: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Обрабатывает данные из файла.
        
        Args:
            filepath: Путь к CSV файлу
            
        Returns:
            Кортеж (X, y) с данными
        """
        if self.logger:
            self.logger.info(f"Начало обработки файла: {filepath}")
        
        # Строим цепочку обработчиков
        chain = self._build_chain()
        
        # Подготавливаем начальные данные
        initial_data = {
            'filepath': filepath,
            'logger': self.logger,
            'config': self.config
        }
        
        # Запускаем цепочку обработки
        result = chain.handle(initial_data)
        
        if result is None:
            raise RuntimeError("Обработка данных завершилась с ошибкой")
        
        # Извлекаем финальные данные
        X_final = result.get('X_final')
        y_final = result.get('y_final')
        
        if X_final is None:
            raise ValueError("Не удалось получить X из пайплайна")
        
        # Если y не создан, создаем заглушку
        if y_final is None or (hasattr(y_final, 'shape') and y_final.shape[0] == 0):
            y_final = np.zeros(X_final.shape[0], dtype=np.float32)
            if self.logger:
                self.logger.warning("Создана заглушка для y (целевая переменная не найдена)")
        
        # Создаем именованные массивы для удобства
        class NamedArray(np.ndarray):
            """Класс для numpy массива с методом save."""
            
            def save(self, path):
                """Сохраняет массив в файл."""
                np.save(path, self)
        
        X_named = X_final.view(NamedArray)
        y_named = y_final.view(NamedArray)
        
        return X_named, y_named