"""
Конкретные обработчики данных для пайплайна.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder, RobustScaler
from sklearn.impute import SimpleImputer
import warnings
import re

from .base_handler import DataHandler


class ChunkedDataLoaderHandler(DataHandler):
    """Оптимизированная загрузка больших HH файлов по частям."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("ChunkedDataLoader", config)
        
    def _process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        filepath = data['filepath']
        chunksize = self.config.get('data', {}).get('chunksize', 50000)
        use_chunks = self.config.get('data', {}).get('use_chunks', True)
        
        if not use_chunks:
            # Возвращаемся к простой загрузке для маленьких файлов
            return self._simple_load(data)
        
        if data.get('logger'):
            data['logger'].info(f"Начинаем потоковую загрузку (chunksize={chunksize})")
        
        # Оптимальные типы для HH данных
        dtypes = {
            'Unnamed: 0': 'int32',
            'Пол, возраст': 'str',
            'ЗП': 'str',
            'Ищет работу на должность:': 'str',
            'Город': 'category',
            'Занятость': 'category',
            'График': 'category',
            'Опыт (двойное нажатие для полной версии)': 'str',
            'Последенее/нынешнее место работы': 'str',
            'Последеняя/нынешняя должность': 'str',
            'Образование и ВУЗ': 'str',
            'Обновление резюме': 'str',
            'Авто': 'category'
        }
        
        chunks = []
        total_rows = 0
        
        # Определяем кодировку
        encodings = ['utf-8', 'cp1251', 'latin1']
        encoding_success = False
        
        for encoding in encodings:
            try:
                # Читаем первую строку для проверки кодировки
                with open(filepath, 'r', encoding=encoding) as f:
                    f.readline()
                data['encoding'] = encoding
                encoding_success = True
                break
            except:
                continue
        
        if not encoding_success:
            data['encoding'] = 'utf-8'
        
        # Читаем по частям
        for i, chunk in enumerate(pd.read_csv(
            filepath,
            dtype=dtypes,
            chunksize=chunksize,
            low_memory=False,
            on_bad_lines='skip',
            encoding=data['encoding']
        )):
            chunks.append(chunk)
            total_rows += len(chunk)
            
            # Логируем прогресс каждые 5 чанков
            if data.get('logger') and (i + 1) % 5 == 0:
                data['logger'].info(f"  Загружено {i + 1} чанков, {total_rows} строк")
        
        # Объединяем все части
        df = pd.concat(chunks, ignore_index=True)
        
        if data.get('logger'):
            data['logger'].info(f"Загрузка завершена. Всего строк: {len(df)}")
            data['logger'].info(f"Размер DataFrame: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
        
        data['dataframe'] = df
        data['original_shape'] = df.shape
        data['loaded_in_chunks'] = True
        
        return data
    
    def _simple_load(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Простая загрузка для маленьких файлов."""
        filepath = data['filepath']
        
        dtypes = {
            'Unnamed: 0': 'int32',
            'Пол, возраст': 'str',
            'ЗП': 'str',
            'Ищет работу на должность:': 'str',
            'Город': 'category',
            'Занятость': 'category',
            'График': 'category',
            'Авто': 'category'
        }
        
        df = pd.read_csv(
            filepath,
            dtype=dtypes,
            low_memory=False,
            on_bad_lines='skip'
        )
        
        data['dataframe'] = df
        data['original_shape'] = df.shape
        data['loaded_in_chunks'] = False
        
        return data


class DataCleanerHandler(DataHandler):
    """Обработчик для очистки HH данных."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("DataCleaner", config)
        
    def _process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        df = data['dataframe']
        config = self.config.get('cleaning', {})
        
        if data.get('logger'):
            data['logger'].info("Начало очистки данных")
        
        # 1. Удаление индексной колонки если есть
        if 'Unnamed: 0' in df.columns:
            df = df.drop(columns=['Unnamed: 0'])
            if data.get('logger'):
                data['logger'].info("Удалена индексная колонка 'Unnamed: 0'")
        
        # 2. Удаление дубликатов
        initial_rows = len(df)
        df = df.drop_duplicates()
        removed_duplicates = initial_rows - len(df)
        
        if removed_duplicates > 0 and data.get('logger'):
            data['logger'].info(f"Удалено дубликатов: {removed_duplicates}")
        
        # 3. Удаление указанных колонок
        drop_columns = config.get('drop_columns', [])
        if drop_columns:
            existing_columns = [col for col in drop_columns if col in df.columns]
            df = df.drop(columns=existing_columns)
            if data.get('logger') and existing_columns:
                data['logger'].info(f"Удалены колонки: {existing_columns}")
        
        # 4. Обработка пропущенных значений
        fillna_config = config.get('fillna', {})
        
        # Для числовых колонок (если появятся после парсинга)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0 and fillna_config.get('numeric'):
            strategy = fillna_config['numeric']
            if isinstance(strategy, (int, float)):
                df[numeric_cols] = df[numeric_cols].fillna(strategy)
            elif strategy in ['mean', 'median']:
                imputer = SimpleImputer(strategy=strategy)
                df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
        
        # Для категориальных колонок
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        if len(categorical_cols) > 0 and fillna_config.get('categorical'):
            fill_value = fillna_config['categorical']
            df[categorical_cols] = df[categorical_cols].fillna(fill_value)
        
        data['dataframe'] = df
        data['cleaning_stats'] = {
            'removed_duplicates': removed_duplicates,
            'remaining_rows': len(df),
            'remaining_columns': len(df.columns)
        }
        
        return data


class HHDataParserHandler(DataHandler):
    """Специальный парсер для HH данных."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("HHDataParser", config)
        
    def _process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        df = data['dataframe']
        
        if data.get('logger'):
            data['logger'].info("Парсинг специфичных полей HH")
        
        # 1. Парсим зарплату
        if 'ЗП' in df.columns:
            df = self._parse_salary(df, data.get('logger'))
        
        # 2. Парсим пол и возраст
        if 'Пол, возраст' in df.columns:
            df = self._parse_age_gender(df, data.get('logger'))
        
        # 3. Обрабатываем длинные текстовые поля
        df = self._handle_text_fields(df, data.get('logger'))
        
        data['dataframe'] = df
        return data
    
    def _parse_salary(self, df: pd.DataFrame, logger=None) -> pd.DataFrame:
        """Парсит колонку с зарплатой."""
        def extract_salary(text):
            if pd.isna(text) or not isinstance(text, str):
                return np.nan
            
            text = str(text).lower()
            
            # Проверяем "не указана"
            if any(word in text for word in ['не указан', 'не указана', 'по договорённости', 'договорная']):
                return np.nan
            
            # Удаляем все нецифровые символы кроме точки, запятой и дефиса
            cleaned = re.sub(r'[^\d\.,\-]', '', text)
            
            # Ищем числа
            numbers = re.findall(r'\d+[\.,]?\d*', cleaned)
            
            if not numbers:
                return np.nan
            
            # Конвертируем в float
            try:
                # Если диапазон (100000-150000), берем среднее
                if '-' in cleaned:
                    parts = cleaned.split('-')
                    nums = []
                    for part in parts:
                        num = part.replace(',', '.')
                        if num:
                            nums.append(float(num))
                    if nums:
                        return sum(nums) / len(nums)
                
                # Иначе берем первое число
                num = numbers[0].replace(',', '.')
                return float(num)
            except:
                return np.nan
        
        original_count = len(df)
        df['ЗП_число'] = df['ЗП'].apply(extract_salary)
        
        parsed_count = df['ЗП_число'].notna().sum()
        parsed_pct = (parsed_count / original_count) * 100
        
        if logger:
            logger.info(f"Зарплата распарсена: {parsed_count}/{original_count} ({parsed_pct:.1f}%)")
            if parsed_count > 0:
                avg = df['ЗП_число'].mean()
                median = df['ЗП_число'].median()
                logger.info(f"Средняя: {avg:,.0f} руб., Медиана: {median:,.0f} руб.")
        
        # Удаляем оригинальную текстовую колонку
        df = df.drop(columns=['ЗП'])
        # Переименовываем в целевую переменную
        df = df.rename(columns={'ЗП_число': 'ЗП'})
        
        return df
    
    def _parse_age_gender(self, df: pd.DataFrame, logger=None) -> pd.DataFrame:
        """Парсит колонку 'Пол, возраст'."""
        def parse_age_gender_cell(text):
            if pd.isna(text) or not isinstance(text, str):
                return None, None
            
            # Извлекаем пол
            gender = None
            text_lower = text.lower()
            if any(word in text_lower for word in ['муж', 'male', 'м']):
                gender = 'М'
            elif any(word in text_lower for word in ['жен', 'female', 'ж']):
                gender = 'Ж'
            
            # Извлекаем возраст
            age = None
            age_match = re.search(r'(\d{1,2})\s*(?:лет|года|год)?', text)
            if age_match:
                try:
                    age = int(age_match.group(1))
                except:
                    pass
            
            return gender, age
        
        # Применяем парсинг
        parsed = df['Пол, возраст'].apply(parse_age_gender_cell)
        df['Пол'] = [p[0] for p in parsed]
        df['Возраст'] = [p[1] for p in parsed]
        
        # Удаляем оригинальную колонку
        df = df.drop(columns=['Пол, возраст'])
        
        if logger:
            age_parsed = df['Возраст'].notna().sum()
            gender_parsed = df['Пол'].notna().sum()
            logger.info(f"Распарсено: возраст {age_parsed}, пол {gender_parsed}")
        
        return df
    
    def _handle_text_fields(self, df: pd.DataFrame, logger=None) -> pd.DataFrame:
        """Обрабатывает длинные текстовые поля."""
        config = self.config.get('features', {})
        max_length = config.get('max_text_length', 100)
        drop_long = config.get('drop_long_text', True)
        
        text_columns = [
            'Опыт (двойное нажатие для полной версии)',
            'Последенее/нынешнее место работы',
            'Последеняя/нынешняя должность',
            'Образование и ВУЗ',
            'Обновление резюме',
            'Ищет работу на должность:'
        ]
        
        if drop_long:
            # Удаляем длинные текстовые колонки
            cols_to_drop = [col for col in text_columns if col in df.columns]
            df = df.drop(columns=cols_to_drop)
            
            if logger and cols_to_drop:
                logger.info(f"Удалены длинные текстовые колонки: {cols_to_drop}")
        else:
            # Усекаем длинные тексты
            for col in text_columns:
                if col in df.columns:
                    df[col] = df[col].astype(str).str.slice(0, max_length)
        
        return df


class FeatureEngineeringHandler(DataHandler):
    """Упрощенный обработчик для инженерии признаков."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("FeatureEngineering", config)
        
    def _process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        df = data['dataframe']
        
        if data.get('logger'):
            data['logger'].info("Начало инженерии признаков")
        
        # 1. Обработка категориальных признаков
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        
        if len(categorical_cols) > 0:
            encoding_method = self.config.get('encoding', {}).get('categorical_method', 'onehot')
            min_frequency = self.config.get('encoding', {}).get('min_frequency', 0.01)
            
            if encoding_method == 'onehot':
                # Упрощенный one-hot encoding
                for col in categorical_cols:
                    # Удаляем редкие категории
                    value_counts = df[col].value_counts(normalize=True)
                    frequent_categories = value_counts[value_counts >= min_frequency].index
                    
                    # Заменяем редкие на 'Другое'
                    df[col] = df[col].where(df[col].isin(frequent_categories), 'Другое')
                
                # Применяем one-hot encoding
                df = pd.get_dummies(
                    df,
                    columns=categorical_cols,
                    prefix_sep='_',
                    drop_first=True,
                    dtype=np.float32  # Для экономии памяти
                )
            elif encoding_method == 'label':
                # Label encoding
                for col in categorical_cols:
                    le = LabelEncoder()
                    df[col] = le.fit_transform(df[col].astype(str))
        
        # 2. Обработка пропусков после кодирования
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            df[numeric_cols] = df[numeric_cols].fillna(0)
        
        if data.get('logger'):
            data['logger'].info(f"После инженерии признаков: {len(df.columns)} колонок")
        
        data['dataframe'] = df
        data['feature_engineering'] = {
            'categorical_encoded': len(categorical_cols),
            'total_features': len(df.columns)
        }
        
        return data


class DataSplitterHandler(DataHandler):
    """Обработчик для разделения данных на признаки и целевую переменную."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("DataSplitter", config)
        
    def _process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        df = data['dataframe']
        target_column = self.config.get('data', {}).get('target_column', 'ЗП')
        
        # Проверяем, есть ли целевая колонка
        if target_column not in df.columns:
            if data.get('logger'):
                data['logger'].warning(
                    f"Целевая колонка '{target_column}' не найдена. "
                    f"Создаем только признаки (X)."
                )
            
            # Используем все колонки как признаки
            X = df.copy()
            y = np.zeros(len(df))  # Заглушка
            data['target_found'] = False
        else:
            # Разделяем на X и y
            X = df.drop(columns=[target_column])
            y = df[target_column].values
            data['target_found'] = True
        
        # Преобразуем в numpy для скорости
        if isinstance(X, pd.DataFrame):
            X_values = X.values
        else:
            X_values = X
        
        data['X'] = X_values
        data['y'] = y
        data['target_column'] = target_column
        
        if data.get('logger'):
            y_shape = y.shape if hasattr(y, 'shape') else len(y)
            data['logger'].info(
                f"Данные разделены. X shape: {X_values.shape}, y shape: {y_shape}"
            )
        
        return data


class ScalerHandler(DataHandler):
    """Упрощенный обработчик для масштабирования признаков."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("Scaler", config)
        
    def _process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        X = data['X']
        scaling_config = self.config.get('scaling', {})
        method = scaling_config.get('method', 'robust')
        
        if data.get('logger'):
            data['logger'].info(f"Применение масштабирования: {method}")
        
        # Быстрое масштабирование в зависимости от метода
        if method == 'robust' and X.shape[1] > 0:
            scaler = RobustScaler()
            X_scaled = scaler.fit_transform(X)
            data['scaler'] = scaler
        elif method == 'standard' and X.shape[1] > 0:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            data['scaler'] = scaler
        elif method == 'minmax' and X.shape[1] > 0:
            from sklearn.preprocessing import MinMaxScaler
            scaler = MinMaxScaler()
            X_scaled = scaler.fit_transform(X)
            data['scaler'] = scaler
        else:
            X_scaled = X  # Без масштабирования
        
        data['X'] = X_scaled
        data['scaling_method'] = method
        
        return data


class FinalizerHandler(DataHandler):
    """Оптимизированный финализирующий обработчик."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("Finalizer", config)
        
    def _process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        # Преобразуем все в numpy arrays с оптимальными типами
        X = data['X']
        y = data['y']
        
        # Убеждаемся, что типы данных оптимальны
        if X.dtype != np.float32:
            X = X.astype(np.float32)
        
        # Обрабатываем целевую переменную
        if hasattr(y, 'dtype') and y.dtype != np.float32:
            y = y.astype(np.float32)
        elif isinstance(y, list) or (hasattr(y, 'shape') and len(y.shape) == 1):
            y = np.array(y, dtype=np.float32)
        
        data['X_final'] = X
        data['y_final'] = y
        
        # Логируем финальную статистику
        if data.get('logger'):
            logger = data['logger']
            logger.info("=" * 50)
            logger.info("ФИНАЛЬНАЯ СТАТИСТИКА:")
            logger.info(f"  Признаки (X): shape={X.shape}, dtype={X.dtype}")
            
            y_size = y.shape if hasattr(y, 'shape') else len(y)
            logger.info(f"  Целевая переменная (y): size={y_size}, dtype={y.dtype}")
            
            if hasattr(X, 'shape'):
                logger.info(f"  Пропущенные значения в X: {np.isnan(X).sum()}")
            
            if hasattr(y, 'shape'):
                logger.info(f"  Пропущенные значения в y: {np.isnan(y).sum()}")
            
            # Размер в памяти
            x_mb = X.nbytes / (1024**2) if hasattr(X, 'nbytes') else 0
            logger.info(f"  Размер X в памяти: {x_mb:.1f} MB")
            logger.info("=" * 50)
        
        return data