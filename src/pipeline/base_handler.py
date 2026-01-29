"""
Базовые классы для реализации паттерна "Цепочка ответственности".
"""

from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import Optional, Any, Dict


class Handler(ABC):
    """
    Абстрактный базовый класс обработчика в цепочке ответственности.
    """
    
    def __init__(self):
        self._next_handler: Optional['Handler'] = None
        
    def set_next(self, handler: 'Handler') -> 'Handler':
        """
        Устанавливает следующий обработчик в цепочке.
        
        Args:
            handler: Следующий обработчик
            
        Returns:
            Следующий обработчик (для fluent interface)
        """
        self._next_handler = handler
        return handler
    
    @abstractmethod
    def handle(self, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Обрабатывает данные. Если не может обработать, передает следующему.
        
        Args:
            data: Словарь с данными и метаинформацией
            
        Returns:
            Обработанные данные или None
        """
        pass
    
    def _pass_to_next(self, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Передает данные следующему обработчику в цепочке.
        
        Args:
            data: Данные для обработки
            
        Returns:
            Результат обработки или None
        """
        if self._next_handler:
            return self._next_handler.handle(data)
        return data


class DataHandler(Handler):
    """
    Базовый класс для обработчиков данных с общими утилитами.
    """
    
    def __init__(self, name: str, config: Dict[str, Any]):
        """
        Инициализирует обработчик данных.
        
        Args:
            name: Имя обработчика для логирования
            config: Конфигурация обработчика
        """
        super().__init__()
        self.name = name
        self.config = config
        
    def handle(self, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Базовая реализация с логированием.
        
        Args:
            data: Словарь с данными
            
        Returns:
            Обработанные данные
        """
        logger = data.get('logger')
        if logger:
            logger.info(f"Обработчик '{self.name}': начало работы")
        
        try:
            result = self._process(data)
            
            if logger:
                logger.info(f"Обработчик '{self.name}': завершено успешно")
            
            # Передаем следующему обработчику
            return self._pass_to_next(result)
            
        except Exception as e:
            if logger:
                logger.error(f"Обработчик '{self.name}': ошибка - {e}")
            raise
    
    @abstractmethod
    def _process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Конкретная реализация обработки данных.
        
        Args:
            data: Словарь с данными
            
        Returns:
            Обработанные данные
        """
        pass