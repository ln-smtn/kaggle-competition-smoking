"""
Pipeline для предсказания курительного статуса пациента
Бинарная классификация с использованием LightGBM и ансамблирования

__author__ = 'Adapted for Smoking Classification'
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.feature_selection import (
    VarianceThreshold, SelectKBest, mutual_info_classif, 
    SelectFromModel, RFE
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier, ElasticNet
import warnings
warnings.filterwarnings('ignore')

# Опциональные импорты
try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print("Optuna не установлена. Установите: pip install optuna")

try:
    from boruta import BorutaPy
    BORUTA_AVAILABLE = True
except ImportError:
    BORUTA_AVAILABLE = False
    print("Boruta не установлена. Установите: pip install Boruta")


# ============================================================================
# DATA COLLECTION & SPLIT
# ============================================================================

def get_input():
    """Загрузка и подготовка данных"""
    train = pd.read_csv('playground-series-s3e24/train.csv')
    test = pd.read_csv('playground-series-s3e24/test.csv')
    
    # Сохранение id для submission
    train_ids = train['id'].copy()
    test_ids = test['id'].copy()
    
    # Выделение целевой переменной
    y_train = train['smoking'].copy()
    
    # Удаление id и целевой переменной из признаков
    X_train = train.drop(['id', 'smoking'], axis=1)
    X_test = test.drop(['id'], axis=1)
    
    return X_train.values, y_train.values, X_test.values, test_ids.values


# ============================================================================
# FEATURE ENGINEERING
# ============================================================================

class FeatureEngineeringTransformer(BaseEstimator, TransformerMixin):
    """Генерация новых признаков"""
    
    def __init__(self):
        self.feature_names_ = None
        self.original_cols_ = [
            'age', 'height(cm)', 'weight(kg)', 'waist(cm)', 'eyesight(left)', 
            'eyesight(right)', 'hearing(left)', 'hearing(right)', 'systolic', 
            'relaxation', 'fasting blood sugar', 'Cholesterol', 'triglyceride', 
            'HDL', 'LDL', 'hemoglobin', 'Urine protein', 'serum creatinine', 
            'AST', 'ALT', 'Gtp', 'dental caries'
        ]
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        """Применение генерации признаков"""
        X = np.array(X) if not isinstance(X, np.ndarray) else X
        
        # Восстанавливаем названия колонок
        cols_to_use = self.original_cols_[:X.shape[1]]
        df = pd.DataFrame(X, columns=cols_to_use)
        
        # Базовые признаки
        df = self._create_basic_features(df)
        
        # Расширенные признаки
        df = self._create_advanced_features(df)
        
        self.feature_names_ = df.columns.tolist()
        return df.values
    
    def _create_basic_features(self, df):
        """Создание базовых признаков"""
        df = df.copy()
        
        # BMI (Body Mass Index)
        if 'height(cm)' in df.columns and 'weight(kg)' in df.columns:
            df['BMI'] = df['weight(kg)'] / ((df['height(cm)'] / 100) ** 2)
        
        # Отношение талии к росту
        if 'waist(cm)' in df.columns and 'height(cm)' in df.columns:
            df['WHR'] = df['waist(cm)'] / df['height(cm)']
        
        # Среднее зрение
        if 'eyesight(left)' in df.columns and 'eyesight(right)' in df.columns:
            df['eyesight_mean'] = (df['eyesight(left)'] + df['eyesight(right)']) / 2
            df['eyesight_diff'] = abs(df['eyesight(left)'] - df['eyesight(right)'])
        
        # Средний слух
        if 'hearing(left)' in df.columns and 'hearing(right)' in df.columns:
            df['hearing_mean'] = (df['hearing(left)'] + df['hearing(right)']) / 2
            df['hearing_diff'] = abs(df['hearing(left)'] - df['hearing(right)'])
        
        # Пульсовое давление
        if 'systolic' in df.columns and 'relaxation' in df.columns:
            df['pulse_pressure'] = df['systolic'] - df['relaxation']
            df['mean_arterial_pressure'] = df['relaxation'] + (df['pulse_pressure'] / 3)
        
        # Отношения холестерина
        if 'HDL' in df.columns and 'LDL' in df.columns:
            df['HDL_LDL_ratio'] = df['HDL'] / (df['LDL'] + 1e-6)
            df['total_cholesterol'] = df['HDL'] + df['LDL'] + (df.get('Cholesterol', 0) / 2)
        
        # Отношение триглицеридов к HDL
        if 'triglyceride' in df.columns and 'HDL' in df.columns:
            df['triglyceride_HDL_ratio'] = df['triglyceride'] / (df['HDL'] + 1e-6)
        
        # Отношение AST к ALT
        if 'AST' in df.columns and 'ALT' in df.columns:
            df['AST_ALT_ratio'] = df['AST'] / (df['ALT'] + 1e-6)
            df['liver_enzymes_sum'] = df['AST'] + df['ALT'] + df.get('Gtp', 0)
        
        # Возрастные группы
        if 'age' in df.columns:
            df['age_group'] = pd.cut(df['age'], bins=[0, 30, 40, 50, 60, 100], 
                                     labels=[0, 1, 2, 3, 4]).astype(int)
        
        return df
    
    def _create_advanced_features(self, df):
        """Создание расширенных признаков"""
        df = df.copy()
        
        # Только самые важные признаки для полиномиальных преобразований
        # Фокус на признаках, которые наиболее связаны с курением
        top_features = ['age', 'BMI', 'systolic', 'hemoglobin', 'Cholesterol', 'HDL', 'LDL']
        
        for feat in top_features:
            if feat in df.columns:
                # Квадрат признака (для нелинейных зависимостей)
                df[f'{feat}_squared'] = df[feat] ** 2
                # Логарифм (для положительных значений, нормализует распределение)
                if (df[feat] > 0).all():
                    df[f'{feat}_log'] = np.log1p(df[feat])
        
        # Только самые важные взаимодействия (клинически значимые)
        # Избегаем создания всех возможных комбинаций
        important_interactions = [
            ('age', 'BMI'),           # Возраст и вес - важный фактор
            ('age', 'systolic'),      # Возраст и давление
            ('BMI', 'hemoglobin'),    # Вес и гемоглобин
            ('HDL', 'LDL'),           # Хороший и плохой холестерин
            ('systolic', 'relaxation'), # Систолическое и диастолическое давление
            ('AST', 'ALT'),           # Ферменты печени
        ]
        
        for feat1, feat2 in important_interactions:
            if feat1 in df.columns and feat2 in df.columns:
                # Умножение (взаимодействие)
                df[f'{feat1}_x_{feat2}'] = df[feat1] * df[feat2]
                # Деление (отношение)
                df[f'{feat1}_div_{feat2}'] = df[feat1] / (df[feat2] + 1e-6)
        
        # Заполнение NaN значений медианой
        df = df.fillna(df.median())
        
        return df


# ============================================================================
# FEATURE TREATMENT (Preprocessing)
# ============================================================================

class OutlierTreatmentTransformer(BaseEstimator, TransformerMixin):
    """Обработка выбросов методом IQR (5-95 перцентили)"""
    
    def __init__(self):
        self.lower_bounds_ = None
        self.upper_bounds_ = None
    
    def fit(self, X, y=None):
        X = np.array(X) if not isinstance(X, np.ndarray) else X
        self.lower_bounds_ = []
        self.upper_bounds_ = []
        
        for col_idx in range(X.shape[1]):
            col_data = X[:, col_idx]
            Q1 = np.percentile(col_data, 5)  # Используем 5-95 перцентили
            Q3 = np.percentile(col_data, 95)
            IQR = Q3 - Q1
            self.lower_bounds_.append(Q1 - 1.5 * IQR)
            self.upper_bounds_.append(Q3 + 1.5 * IQR)
        
        return self
    
    def transform(self, X):
        X = np.array(X) if not isinstance(X, np.ndarray) else X.copy()
        X = X.copy()
        
        for col_idx in range(X.shape[1]):
            X[:, col_idx] = np.clip(
                X[:, col_idx],
                self.lower_bounds_[col_idx],
                self.upper_bounds_[col_idx]
            )
        
        return X


class MissingValueTreatmentTransformer(BaseEstimator, TransformerMixin):
    """Обработка пропущенных значений"""
    
    def __init__(self):
        self.median_values_ = None
    
    def fit(self, X, y=None):
        X = np.array(X) if not isinstance(X, np.ndarray) else X
        self.median_values_ = []
        
        for col_idx in range(X.shape[1]):
            col_data = X[:, col_idx]
            # Используем медиану для заполнения пропусков
            median_val = np.nanmedian(col_data)
            self.median_values_.append(median_val)
        
        return self
    
    def transform(self, X):
        X = np.array(X) if not isinstance(X, np.ndarray) else X.copy()
        X = X.copy()
        
        # Заполнение пропусков медианой
        for col_idx in range(X.shape[1]):
            col_data = X[:, col_idx]
            mask = np.isnan(col_data)
            if mask.any():
                X[mask, col_idx] = self.median_values_[col_idx]
        
        return X


# ============================================================================
# FEATURE JITTERING (Data Augmentation)
# ============================================================================

class FeatureJitteringTransformer(BaseEstimator, TransformerMixin):
    """Добавление небольшого шума к числовым признакам (только для train)"""
    
    def __init__(self, noise_std=0.01, apply_to_train=True):
        self.noise_std = noise_std
        self.apply_to_train = apply_to_train
        self.is_training_ = False
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        if not self.apply_to_train or not self.is_training_:
            return X
        
        X = np.array(X) if not isinstance(X, np.ndarray) else X.copy()
        X = X.copy()
        
        # Добавляем небольшой гауссовский шум
        noise = np.random.normal(0, self.noise_std, X.shape)
        X = X + noise
        
        return X
    
    def set_training(self, is_training=True):
        """Установить режим обучения"""
        self.is_training_ = is_training


# ============================================================================
# FEATURE TRANSFORM
# ============================================================================

class StandartScalerTransformer(BaseEstimator, TransformerMixin):
    """Масштабирование признаков с помощью StandardScaler"""
    
    def __init__(self):
        self.scaler_ = StandardScaler()
    
    def fit(self, X, y=None):
        self.scaler_.fit(X)
        return self
    
    def transform(self, X):
        return self.scaler_.transform(X)


# ============================================================================
# FEATURE SELECTION - РАСШИРЕННЫЙ ОТБОР ПРИЗНАКОВ
# ============================================================================

class VarianceThresholdTransformer(BaseEstimator, TransformerMixin):
    """Удаление признаков с низкой дисперсией"""
    
    def __init__(self, threshold=0.0):
        self.vt_ = VarianceThreshold(threshold=threshold)
    
    def fit(self, X, y=None):
        self.vt_.fit(X)
        return self
    
    def transform(self, X):
        return self.vt_.transform(X)


class CorrelationFilterTransformer(BaseEstimator, TransformerMixin):
    """Быстрый фильтр - оставляем фичи с non-0 корреляцией с таргетом"""
    
    def __init__(self, threshold=0.01):
        self.threshold = threshold
        self.selected_features_ = None
    
    def fit(self, X, y):
        X = np.array(X) if not isinstance(X, np.ndarray) else X
        y = np.array(y) if not isinstance(y, np.ndarray) else y
        
        # Вычисляем корреляции с таргетом
        correlations = []
        for i in range(X.shape[1]):
            corr = np.abs(np.corrcoef(X[:, i], y)[0, 1])
            correlations.append(corr)
        
        correlations = np.array(correlations)
        # Оставляем признаки с корреляцией выше порога
        self.selected_features_ = np.where(correlations > self.threshold)[0]
        
        return self
    
    def transform(self, X):
        X = np.array(X) if not isinstance(X, np.ndarray) else X
        return X[:, self.selected_features_]


class MutualInformationTransformer(BaseEstimator, TransformerMixin):
    """Отбор признаков на основе Mutual Information"""
    
    def __init__(self, k=50):
        self.k = k
        self.selector_ = None
    
    def fit(self, X, y):
        self.selector_ = SelectKBest(score_func=mutual_info_classif, k=self.k)
        self.selector_.fit(X, y)
        return self
    
    def transform(self, X):
        return self.selector_.transform(X)


class StableFeatureImportanceSelector(BaseEstimator, TransformerMixin):
    """Отбор признаков на основе стабильной Feature Importance на разных seed"""
    
    def __init__(self, n_seeds=5, top_k=50, cv=5):
        self.n_seeds = n_seeds
        self.top_k = top_k
        self.cv = cv
        self.selected_features_ = None
        self.importance_scores_ = None
    
    def fit(self, X, y):
        X = np.array(X) if not isinstance(X, np.ndarray) else X
        y = np.array(y) if not isinstance(y, np.ndarray) else y
        
        n_features = X.shape[1]
        importance_matrix = np.zeros((self.n_seeds, n_features))
        
        # Вычисляем важность признаков на разных seed
        for seed in range(self.n_seeds):
            skf = StratifiedKFold(n_splits=self.cv, shuffle=True, random_state=seed)
            fold_importances = []
            
            for train_idx, val_idx in skf.split(X, y):
                X_tr, X_val = X[train_idx], X[val_idx]
                y_tr, y_val = y[train_idx], y[val_idx]
                
                # Обучаем модель для получения важности
                model = lgb.LGBMClassifier(
                    n_estimators=100,
                    random_state=seed,
                    verbose=-1
                )
                model.fit(X_tr, y_tr)
                fold_importances.append(model.feature_importances_)
            
            # Усредняем важность по фолдам
            importance_matrix[seed] = np.mean(fold_importances, axis=0)
        
        # Усредняем важность по seed (стабильность)
        self.importance_scores_ = np.mean(importance_matrix, axis=0)
        
        # Отбираем топ-k признаков
        top_indices = np.argsort(self.importance_scores_)[-self.top_k:]
        self.selected_features_ = np.sort(top_indices)
        
        return self
    
    def transform(self, X):
        X = np.array(X) if not isinstance(X, np.ndarray) else X
        return X[:, self.selected_features_]


class PermutationImportanceSelector(BaseEstimator, TransformerMixin):
    """Отбор признаков на основе Permutation Importance (мешаем значения признака)"""
    
    def __init__(self, n_iterations=5, threshold=0.01, cv=5):
        self.n_iterations = n_iterations
        self.threshold = threshold
        self.cv = cv
        self.selected_features_ = None
        self.importance_scores_ = None
    
    def fit(self, X, y):
        X = np.array(X) if not isinstance(X, np.ndarray) else X
        y = np.array(y) if not isinstance(y, np.ndarray) else y
        
        n_features = X.shape[1]
        importance_scores = np.zeros(n_features)
        
        skf = StratifiedKFold(n_splits=self.cv, shuffle=True, random_state=42)
        
        # Базовый score без перестановки
        base_scores = []
        for train_idx, val_idx in skf.split(X, y):
            X_tr, X_val = X[train_idx], X[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]
            
            model = lgb.LGBMClassifier(n_estimators=100, verbose=-1, random_state=42)
            model.fit(X_tr, y_tr)
            base_score = roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])
            base_scores.append(base_score)
        
        base_score_mean = np.mean(base_scores)
        
        # Для каждого признака мешаем значения и меряем изменение
        for feat_idx in range(n_features):
            perm_scores = []
            
            for _ in range(self.n_iterations):
                X_perm = X.copy()
                np.random.shuffle(X_perm[:, feat_idx])  # Мешаем значения признака
                
                fold_scores = []
                for train_idx, val_idx in skf.split(X_perm, y):
                    X_tr, X_val = X_perm[train_idx], X_perm[val_idx]
                    y_tr, y_val = y[train_idx], y[val_idx]
                    
                    model = lgb.LGBMClassifier(n_estimators=100, verbose=-1, random_state=42)
                    model.fit(X_tr, y_tr)
                    perm_score = roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])
                    fold_scores.append(perm_score)
                
                perm_scores.append(np.mean(fold_scores))
            
            # Важность = разница между базовым score и score после перестановки
            importance_scores[feat_idx] = base_score_mean - np.mean(perm_scores)
        
        self.importance_scores_ = importance_scores
        
        # Отбираем признаки с важностью выше порога
        self.selected_features_ = np.where(importance_scores > self.threshold)[0]
        
        return self
    
    def transform(self, X):
        X = np.array(X) if not isinstance(X, np.ndarray) else X
        return X[:, self.selected_features_]


class RecursiveFeatureEliminationTransformer(BaseEstimator, TransformerMixin):
    """RFE - обучение с признаком/без признака на cross-validation"""
    
    def __init__(self, n_features_to_select=50, cv=5):
        self.n_features_to_select = n_features_to_select
        self.cv = cv
        self.selector_ = None
    
    def fit(self, X, y):
        estimator = lgb.LGBMClassifier(n_estimators=100, verbose=-1, random_state=42)
        self.selector_ = RFE(
            estimator=estimator,
            n_features_to_select=self.n_features_to_select,
            step=1
        )
        self.selector_.fit(X, y)
        return self
    
    def transform(self, X):
        return self.selector_.transform(X)


class BorutaFeatureSelector(BaseEstimator, TransformerMixin):
    """Отбор признаков с помощью Boruta (если доступна)"""
    
    def __init__(self, max_iter=100, random_state=42):
        self.max_iter = max_iter
        self.random_state = random_state
        self.selector_ = None
    
    def fit(self, X, y):
        if not BORUTA_AVAILABLE:
            raise ImportError("Boruta не установлена. Установите: pip install Boruta")
        
        rf = RandomForestClassifier(n_estimators=100, random_state=self.random_state, n_jobs=-1)
        self.selector_ = BorutaPy(
            rf,
            n_estimators='auto',
            verbose=0,
            max_iter=self.max_iter,
            random_state=self.random_state
        )
        self.selector_.fit(X, y)
        return self
    
    def transform(self, X):
        return self.selector_.transform(X)


# ============================================================================
# MODEL TRAINER WITH CV
# ============================================================================

class LightGBMClassifierCV(BaseEstimator, ClassifierMixin):
    """LightGBM классификатор с кросс-валидацией"""
    
    def __init__(self, lgb_params=None, fit_params=None, cv=5):
        self.lgb_params = lgb_params or {}
        self.fit_params = fit_params or {}
        self.cv = cv
        self.estimators_ = []
        self.cv_scores_ = []
        self.fold_indices_ = []
    
    @property
    def feature_importances_(self):
        """Средняя важность признаков по всем фолдам"""
        if not self.estimators_:
            return None
        feature_importances = []
        for estimator in self.estimators_:
            feature_importances.append(estimator.feature_importance())
        return np.mean(feature_importances, axis=0)
    
    @property
    def cv_score_(self):
        """Средний ROC-AUC по всем фолдам"""
        return np.mean(self.cv_scores_) if self.cv_scores_ else None
    
    def fit(self, X, y, **fit_params):
        """Обучение модели с кросс-валидацией"""
        skf = StratifiedKFold(n_splits=self.cv, shuffle=True, random_state=42)
        self.estimators_ = []
        self.cv_scores_ = []
        self.fold_indices_ = []
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            self.fold_indices_.append((train_idx, val_idx))
            X_tr, X_val = X[train_idx], X[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]
            
            # Создание датасетов LightGBM
            train_data = lgb.Dataset(X_tr, label=y_tr)
            val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
            
            # Обучение модели
            model = lgb.train(
                self.lgb_params,
                train_data,
                valid_sets=[val_data],
                num_boost_round=700,  
                callbacks=[
                    lgb.early_stopping(stopping_rounds=50),  # Останавливается при отсутствии улучшения 50 раундов
                    lgb.log_evaluation(0)
                ],
                **self.fit_params
            )
            
            # Предсказания и оценка
            y_pred = model.predict(X_val, num_iteration=model.best_iteration)
            score = roc_auc_score(y_val, y_pred)
            
            self.estimators_.append(model)
            self.cv_scores_.append(score)
            
            print(f"Fold {fold + 1}: ROC-AUC = {score:.5f}")
        
        print(f"Средний ROC-AUC: {self.cv_score_:.5f} (+/- {np.std(self.cv_scores_):.5f})")
        return self
    
    def predict_proba(self, X):
        """Предсказание вероятностей (усреднение по всем моделям)"""
        if not self.estimators_:
            raise ValueError("Модель не обучена. Вызовите fit() сначала.")
        
        predictions = []
        for estimator in self.estimators_:
            pred = estimator.predict(X, num_iteration=estimator.best_iteration)
            predictions.append(pred)
        
        # Усреднение предсказаний
        mean_pred = np.mean(predictions, axis=0)
        
        # Преобразование в формат predict_proba (две колонки для бинарной классификации)
        return np.column_stack([1 - mean_pred, mean_pred])
    
    def predict(self, X):
        """Предсказание классов"""
        proba = self.predict_proba(X)
        return (proba[:, 1] > 0.5).astype(int)


# ============================================================================
# ENSEMBLING
# ============================================================================

class EnsemblePredictor:
    """Ансамблирование предсказаний от нескольких моделей"""
    
    def __init__(self, models, weights=None):
        self.models = models
        self.weights = weights if weights is not None else np.ones(len(models)) / len(models)
        self.weights = np.array(self.weights) / np.sum(self.weights)  # Нормализация
    
    def predict_proba(self, X):
        """Взвешенное усреднение предсказаний от всех моделей"""
        predictions = []
        for model in self.models:
            if hasattr(model, 'predict_proba'):
                pred_proba = model.predict_proba(X)
                # Проверяем форму: если двумерный массив, берем вероятность положительного класса
                if pred_proba.ndim == 2:
                    pred = pred_proba[:, 1]  # Вероятность положительного класса
                else:
                    # Если одномерный массив, используем как есть
                    pred = pred_proba
            else:
                pred = model.predict(X)
            predictions.append(pred)
        
        predictions = np.array(predictions)
        # Взвешенное усреднение
        weighted_pred = np.average(predictions, axis=0, weights=self.weights)
        return weighted_pred


# ============================================================================
# HYPERPARAMETER OPTIMIZATION (Optuna)
# ============================================================================

class OptunaOptimizer:
    """Оптимизация гиперпараметров с помощью Optuna"""
    
    def __init__(self, model_type='lgb', n_trials=50, cv=5, random_state=42):
        self.model_type = model_type
        self.n_trials = n_trials
        self.cv = cv
        self.random_state = random_state
        self.best_params_ = None
        self.best_score_ = None
        
        if not OPTUNA_AVAILABLE:
            raise ImportError("Optuna не установлена. Установите: pip install optuna")
    
    def _objective(self, trial, X, y):
        """Целевая функция для оптимизации"""
        skf = StratifiedKFold(n_splits=self.cv, shuffle=True, random_state=self.random_state)
        scores = []
        
        if self.model_type == 'lgb':
            params = {
                'objective': 'binary',
                'metric': 'auc',
                'boosting_type': 'gbdt',
                'num_leaves': trial.suggest_int('num_leaves', 20, 100),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),
                'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1.0),
                'bagging_freq': trial.suggest_int('bagging_freq', 1, 10),
                'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 10, 100),
                'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 10.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 10.0),
                'verbose': -1,
                'random_state': self.random_state
            }
            
            for train_idx, val_idx in skf.split(X, y):
                X_tr, X_val = X[train_idx], X[val_idx]
                y_tr, y_val = y[train_idx], y[val_idx]
                
                train_data = lgb.Dataset(X_tr, label=y_tr)
                val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
                
                model = lgb.train(
                    params,
                    train_data,
                    valid_sets=[val_data],
                num_boost_round=700,  # Баланс скорости и качества
                callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(0)]  # Останавливается при отсутствии улучшения
                )
                
                y_pred = model.predict(X_val, num_iteration=model.best_iteration)
                score = roc_auc_score(y_val, y_pred)
                scores.append(score)
        
        elif self.model_type == 'xgb':
            params = {
                'max_depth': trial.suggest_int('max_depth', 3, 15),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'random_state': self.random_state
            }
            
            for train_idx, val_idx in skf.split(X, y):
                X_tr, X_val = X[train_idx], X[val_idx]
                y_tr, y_val = y[train_idx], y[val_idx]
                
                model = xgb.XGBClassifier(
                    n_estimators=1000,
                    early_stopping_rounds=100,
                    eval_metric='auc',
                    **params
                )
                model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
                
                y_pred = model.predict_proba(X_val)[:, 1]
                score = roc_auc_score(y_val, y_pred)
                scores.append(score)
        
        elif self.model_type == 'catboost':
            params = {
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'depth': trial.suggest_int('depth', 4, 10),
                'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1.0, 10.0),
                'random_state': self.random_state,
                'verbose': False
            }
            
            for train_idx, val_idx in skf.split(X, y):
                X_tr, X_val = X[train_idx], X[val_idx]
                y_tr, y_val = y[train_idx], y[val_idx]
                
                model = CatBoostClassifier(
                    iterations=1000,
                    early_stopping_rounds=100,
                    **params
                )
                model.fit(X_tr, y_tr, eval_set=(X_val, y_val), verbose=False)
                
                y_pred = model.predict_proba(X_val)[:, 1]
                score = roc_auc_score(y_val, y_pred)
                scores.append(score)
        
        return np.mean(scores)
    
    def optimize(self, X, y):
        """Оптимизация гиперпараметров"""
        study = optuna.create_study(direction='maximize', study_name=f'{self.model_type}_optimization')
        study.optimize(
            lambda trial: self._objective(trial, X, y),
            n_trials=self.n_trials,
            show_progress_bar=True
        )
        
        self.best_params_ = study.best_params
        self.best_score_ = study.best_value
        
        print(f"\nЛучшие параметры для {self.model_type}:")
        print(self.best_params_)
        print(f"Лучший ROC-AUC: {self.best_score_:.5f}")
        
        return self.best_params_


# ============================================================================
# XGBOOST & CATBOOST WITH CV
# ============================================================================

class XGBoostClassifierCV(BaseEstimator, ClassifierMixin):
    """XGBoost классификатор с кросс-валидацией"""
    
    def __init__(self, xgb_params=None, cv=5, random_state=42):
        self.xgb_params = xgb_params or {}
        self.cv = cv
        self.random_state = random_state
        self.estimators_ = []
        self.cv_scores_ = []
    
    def fit(self, X, y):
        skf = StratifiedKFold(n_splits=self.cv, shuffle=True, random_state=self.random_state)
        self.estimators_ = []
        self.cv_scores_ = []
        
        # Убираем random_state из xgb_params, чтобы избежать дублирования
        xgb_params_clean = {k: v for k, v in self.xgb_params.items() if k != 'random_state'}
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            X_tr, X_val = X[train_idx], X[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]
            
            model = xgb.XGBClassifier(
                n_estimators=1000,
                early_stopping_rounds=100,
                eval_metric='auc',
                random_state=self.random_state,
                **xgb_params_clean
            )
            model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
            
            y_pred = model.predict_proba(X_val)[:, 1]
            score = roc_auc_score(y_val, y_pred)
            
            self.estimators_.append(model)
            self.cv_scores_.append(score)
            
            print(f"XGBoost Fold {fold + 1}: ROC-AUC = {score:.5f}")
        
        print(f"XGBoost Средний ROC-AUC: {np.mean(self.cv_scores_):.5f}")
        return self
    
    def predict_proba(self, X):
        predictions = []
        for estimator in self.estimators_:
            pred = estimator.predict_proba(X)[:, 1]
            predictions.append(pred)
        mean_pred = np.mean(predictions, axis=0)
        return np.column_stack([1 - mean_pred, mean_pred])


class CatBoostClassifierCV(BaseEstimator, ClassifierMixin):
    """CatBoost классификатор с кросс-валидацией"""
    
    def __init__(self, cat_params=None, cv=5, random_state=42):
        self.cat_params = cat_params or {}
        self.cv = cv
        self.random_state = random_state
        self.estimators_ = []
        self.cv_scores_ = []
    
    def fit(self, X, y):
        skf = StratifiedKFold(n_splits=self.cv, shuffle=True, random_state=self.random_state)
        self.estimators_ = []
        self.cv_scores_ = []
        
        # Убираем random_state и verbose из cat_params, чтобы избежать дублирования
        cat_params_clean = {k: v for k, v in self.cat_params.items() if k not in ['random_state', 'verbose']}
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            X_tr, X_val = X[train_idx], X[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]
            
            model = CatBoostClassifier(
                iterations=1000,
                early_stopping_rounds=100,
                random_state=self.random_state,
                verbose=False,
                **cat_params_clean
            )
            model.fit(X_tr, y_tr, eval_set=(X_val, y_val), verbose=False)
            
            y_pred = model.predict_proba(X_val)[:, 1]
            score = roc_auc_score(y_val, y_pred)
            
            self.estimators_.append(model)
            self.cv_scores_.append(score)
            
            print(f"CatBoost Fold {fold + 1}: ROC-AUC = {score:.5f}")
        
        print(f"CatBoost Средний ROC-AUC: {np.mean(self.cv_scores_):.5f}")
        return self
    
    def predict_proba(self, X):
        predictions = []
        for estimator in self.estimators_:
            pred = estimator.predict_proba(X)[:, 1]
            predictions.append(pred)
        mean_pred = np.mean(predictions, axis=0)
        return np.column_stack([1 - mean_pred, mean_pred])


# ============================================================================
# STACKING & META-MODEL
# ============================================================================

class StackingClassifier(BaseEstimator, ClassifierMixin):
    """Стекинг с мета-моделью"""
    
    def __init__(self, base_models, meta_model=None, cv=5, random_state=42):
        self.base_models = base_models
        self.meta_model = meta_model or LogisticRegression(random_state=random_state)
        self.cv = cv
        self.random_state = random_state
        self.fitted_base_models_ = []
        self.fitted_meta_model_ = None
    
    def fit(self, X, y):
        """Обучение стекинга"""
        skf = StratifiedKFold(n_splits=self.cv, shuffle=True, random_state=self.random_state)
        n_samples = X.shape[0]
        n_models = len(self.base_models)
        
        # Матрица мета-признаков
        meta_features = np.zeros((n_samples, n_models))
        
        # Обучаем базовые модели и создаем мета-признаки
        for model_idx, model in enumerate(self.base_models):
            model_predictions = np.zeros(n_samples)
            
            for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
                X_tr, X_val = X[train_idx], X[val_idx]
                y_tr, y_val = y[train_idx], y[val_idx]
                
                # Обучаем модель на train фолде
                if isinstance(model, dict):
                    # Если модель передана как словарь параметров
                    model_class = model['class']
                    model_params = model.get('params', {})
                    model_copy = model_class(**model_params)
                else:
                    # Если модель передана как объект
                    model_copy = type(model)(**model.get_params() if hasattr(model, 'get_params') else {})
                
                # Для CatBoost отключаем verbose вывод
                if isinstance(model_copy, CatBoostClassifier):
                    model_copy.fit(X_tr, y_tr, verbose=False)
                else:
                    model_copy.fit(X_tr, y_tr)
                
                # Предсказания на validation фолде
                if hasattr(model_copy, 'predict_proba'):
                    pred = model_copy.predict_proba(X_val)[:, 1]
                else:
                    pred = model_copy.predict(X_val)
                
                model_predictions[val_idx] = pred
            
            meta_features[:, model_idx] = model_predictions
        
        # Обучаем мета-модель на мета-признаках
        self.fitted_meta_model_ = self.meta_model
        self.fitted_meta_model_.fit(meta_features, y)
        
        # Обучаем базовые модели на всех данных для финальных предсказаний
        self.fitted_base_models_ = []
        for model in self.base_models:
            if isinstance(model, dict):
                model_class = model['class']
                model_params = model.get('params', {})
                model_copy = model_class(**model_params)
            else:
                model_copy = type(model)(**model.get_params() if hasattr(model, 'get_params') else {})
            # Для CatBoost отключаем verbose вывод
            if isinstance(model_copy, CatBoostClassifier):
                model_copy.fit(X, y, verbose=False)
            else:
                model_copy.fit(X, y)
            self.fitted_base_models_.append(model_copy)
        
        return self
    
    def predict_proba(self, X):
        """Предсказания стекинга"""
        # Получаем предсказания от всех базовых моделей
        meta_features = np.zeros((X.shape[0], len(self.fitted_base_models_)))
        
        for model_idx, model in enumerate(self.fitted_base_models_):
            if hasattr(model, 'predict_proba'):
                pred = model.predict_proba(X)[:, 1]
            else:
                pred = model.predict(X)
            meta_features[:, model_idx] = pred
        
        # Предсказания мета-модели
        return self.fitted_meta_model_.predict_proba(meta_features)


# ============================================================================
# HYPERPARAMETER OPTIMIZATION (Optuna)
# ============================================================================

class OptunaOptimizer:
    """Оптимизация гиперпараметров с помощью Optuna"""
    
    def __init__(self, model_type='lgb', n_trials=50, cv=5, random_state=42):
        self.model_type = model_type
        self.n_trials = n_trials
        self.cv = cv
        self.random_state = random_state
        self.best_params_ = None
        self.best_score_ = None
        
        if not OPTUNA_AVAILABLE:
            raise ImportError("Optuna не установлена. Установите: pip install optuna")
    
    def _objective(self, trial, X, y):
        """Целевая функция для оптимизации"""
        skf = StratifiedKFold(n_splits=self.cv, shuffle=True, random_state=self.random_state)
        scores = []
        
        if self.model_type == 'lgb':
            params = {
                'objective': 'binary',
                'metric': 'auc',
                'boosting_type': 'gbdt',
                'num_leaves': trial.suggest_int('num_leaves', 20, 100),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),
                'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1.0),
                'bagging_freq': trial.suggest_int('bagging_freq', 1, 10),
                'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 10, 100),
                'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 10.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 10.0),
                'verbose': -1,
                'random_state': self.random_state
            }
            
            for train_idx, val_idx in skf.split(X, y):
                X_tr, X_val = X[train_idx], X[val_idx]
                y_tr, y_val = y[train_idx], y[val_idx]
                
                train_data = lgb.Dataset(X_tr, label=y_tr)
                val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
                
                model = lgb.train(
                    params,
                    train_data,
                    valid_sets=[val_data],
                num_boost_round=700,  # Баланс скорости и качества
                callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(0)]  # Останавливается при отсутствии улучшения
                )
                
                y_pred = model.predict(X_val, num_iteration=model.best_iteration)
                score = roc_auc_score(y_val, y_pred)
                scores.append(score)
        
        elif self.model_type == 'xgb':
            params = {
                'max_depth': trial.suggest_int('max_depth', 3, 15),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'random_state': self.random_state
            }
            
            for train_idx, val_idx in skf.split(X, y):
                X_tr, X_val = X[train_idx], X[val_idx]
                y_tr, y_val = y[train_idx], y[val_idx]
                
                model = xgb.XGBClassifier(
                    n_estimators=1000,
                    early_stopping_rounds=100,
                    eval_metric='auc',
                    **params
                )
                model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
                
                y_pred = model.predict_proba(X_val)[:, 1]
                score = roc_auc_score(y_val, y_pred)
                scores.append(score)
        
        elif self.model_type == 'catboost':
            params = {
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'depth': trial.suggest_int('depth', 4, 10),
                'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1.0, 10.0),
                'random_state': self.random_state,
                'verbose': False
            }
            
            for train_idx, val_idx in skf.split(X, y):
                X_tr, X_val = X[train_idx], X[val_idx]
                y_tr, y_val = y[train_idx], y[val_idx]
                
                model = CatBoostClassifier(
                    iterations=1000,
                    early_stopping_rounds=100,
                    **params
                )
                model.fit(X_tr, y_tr, eval_set=(X_val, y_val), verbose=False)
                
                y_pred = model.predict_proba(X_val)[:, 1]
                score = roc_auc_score(y_val, y_pred)
                scores.append(score)
        
        return np.mean(scores)
    
    def optimize(self, X, y):
        """Оптимизация гиперпараметров"""
        study = optuna.create_study(direction='maximize', study_name=f'{self.model_type}_optimization')
        study.optimize(
            lambda trial: self._objective(trial, X, y),
            n_trials=self.n_trials,
            show_progress_bar=True
        )
        
        self.best_params_ = study.best_params
        self.best_score_ = study.best_value
        
        print(f"\nЛучшие параметры для {self.model_type}:")
        print(self.best_params_)
        print(f"Лучший ROC-AUC: {self.best_score_:.5f}")
        
        return self.best_params_


# ============================================================================
# XGBOOST & CATBOOST WITH CV
# ============================================================================

class XGBoostClassifierCV(BaseEstimator, ClassifierMixin):
    """XGBoost классификатор с кросс-валидацией"""
    
    def __init__(self, xgb_params=None, cv=5, random_state=42):
        self.xgb_params = xgb_params or {}
        self.cv = cv
        self.random_state = random_state
        self.estimators_ = []
        self.cv_scores_ = []
    
    def fit(self, X, y):
        skf = StratifiedKFold(n_splits=self.cv, shuffle=True, random_state=self.random_state)
        self.estimators_ = []
        self.cv_scores_ = []
        
        # Убираем random_state из xgb_params, чтобы избежать дублирования
        xgb_params_clean = {k: v for k, v in self.xgb_params.items() if k != 'random_state'}
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            X_tr, X_val = X[train_idx], X[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]
            
            model = xgb.XGBClassifier(
                n_estimators=1000,
                early_stopping_rounds=100,
                eval_metric='auc',
                random_state=self.random_state,
                **xgb_params_clean
            )
            model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
            
            y_pred = model.predict_proba(X_val)[:, 1]
            score = roc_auc_score(y_val, y_pred)
            
            self.estimators_.append(model)
            self.cv_scores_.append(score)
            
            print(f"XGBoost Fold {fold + 1}: ROC-AUC = {score:.5f}")
        
        print(f"XGBoost Средний ROC-AUC: {np.mean(self.cv_scores_):.5f}")
        return self
    
    def predict_proba(self, X):
        predictions = []
        for estimator in self.estimators_:
            pred = estimator.predict_proba(X)[:, 1]
            predictions.append(pred)
        mean_pred = np.mean(predictions, axis=0)
        return np.column_stack([1 - mean_pred, mean_pred])


class CatBoostClassifierCV(BaseEstimator, ClassifierMixin):
    """CatBoost классификатор с кросс-валидацией"""
    
    def __init__(self, cat_params=None, cv=5, random_state=42):
        self.cat_params = cat_params or {}
        self.cv = cv
        self.random_state = random_state
        self.estimators_ = []
        self.cv_scores_ = []
    
    def fit(self, X, y):
        skf = StratifiedKFold(n_splits=self.cv, shuffle=True, random_state=self.random_state)
        self.estimators_ = []
        self.cv_scores_ = []
        
        # Убираем random_state и verbose из cat_params, чтобы избежать дублирования
        cat_params_clean = {k: v for k, v in self.cat_params.items() if k not in ['random_state', 'verbose']}
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            X_tr, X_val = X[train_idx], X[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]
            
            model = CatBoostClassifier(
                iterations=1000,
                early_stopping_rounds=100,
                random_state=self.random_state,
                verbose=False,
                **cat_params_clean
            )
            model.fit(X_tr, y_tr, eval_set=(X_val, y_val), verbose=False)
            
            y_pred = model.predict_proba(X_val)[:, 1]
            score = roc_auc_score(y_val, y_pred)
            
            self.estimators_.append(model)
            self.cv_scores_.append(score)
            
            print(f"CatBoost Fold {fold + 1}: ROC-AUC = {score:.5f}")
        
        print(f"CatBoost Средний ROC-AUC: {np.mean(self.cv_scores_):.5f}")
        return self
    
    def predict_proba(self, X):
        predictions = []
        for estimator in self.estimators_:
            pred = estimator.predict_proba(X)[:, 1]
            predictions.append(pred)
        mean_pred = np.mean(predictions, axis=0)
        return np.column_stack([1 - mean_pred, mean_pred])

# ============================================================================
# PSEUDO LABELING
# ============================================================================

class PseudoLabeling:
    """Pseudo-labeling: использование предсказаний на test для расширения train"""
    
    def __init__(self, threshold=0.95, max_samples=None):
        self.threshold = threshold  # Порог уверенности для pseudo-labels
        self.max_samples = max_samples  # Максимальное количество pseudo-labels
    
    def generate_pseudo_labels(self, model, X_test, test_ids):
        """Генерация pseudo-labels на основе предсказаний модели"""
        # Получаем предсказания
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(X_test)[:, 1]
        else:
            proba = model.predict(X_test)
        
        # Выбираем высокоуверенные предсказания
        high_confidence_mask = (proba >= self.threshold) | (proba <= (1 - self.threshold))
        
        pseudo_X = X_test[high_confidence_mask]
        pseudo_y = (proba[high_confidence_mask] > 0.5).astype(int)
        pseudo_ids = test_ids[high_confidence_mask]
        
        # Ограничиваем количество если нужно
        if self.max_samples and len(pseudo_X) > self.max_samples:
            indices = np.random.choice(len(pseudo_X), self.max_samples, replace=False)
            pseudo_X = pseudo_X[indices]
            pseudo_y = pseudo_y[indices]
            pseudo_ids = pseudo_ids[indices]
        
        return pseudo_X, pseudo_y, pseudo_ids


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    """
    СИЛЬНЫЙ BASELINE С LIGHTGBM
    
    Пайплайн:
    1. Data Collection & Split
    2. Feature Engineering (улучшенные признаки)
    3. Feature Transform (стандартизация)
    4. Feature Treatment (пропуски, выбросы)
    5. Feature Selection (удаление шумных признаков, высокий signal2noise)
    6. Model Trainer (LightGBM с оптимизацией гиперпараметров)
    """
    
    # Загрузка данных
    print("\n1. DATA COLLECTION & SPLIT")
    print("-" * 60)
    X_train, y_train, X_test, test_ids = get_input()
    
    print(f"✓ Размер обучающей выборки: {X_train.shape}")
    print(f"✓ Размер тестовой выборки: {X_test.shape}")
    print(f"✓ Размер целевой переменной: {y_train.shape}")
    print(f"✓ Распределение классов: {np.bincount(y_train)}")
    
    # Предобработка данных
    print("\n2. FEATURE ENGINEERING & PREPROCESSING")
    print("-" * 60)
    
    fe_transformer = FeatureEngineeringTransformer()
    X_train_fe = fe_transformer.fit_transform(X_train)
    X_test_fe = fe_transformer.transform(X_test)
    
    print(f"После Feature Engineering: {X_train_fe.shape[1]} признаков")
    
    # Feature Treatment
    mv_transformer = MissingValueTreatmentTransformer()
    ot_transformer = OutlierTreatmentTransformer()
    
    X_train_fe = mv_transformer.fit_transform(X_train_fe)
    X_test_fe = mv_transformer.transform(X_test_fe)
    
    X_train_fe = ot_transformer.fit_transform(X_train_fe)
    X_test_fe = ot_transformer.transform(X_test_fe)
    
    # Feature Transform
    scaler = StandartScalerTransformer()
    X_train_scaled = scaler.fit_transform(X_train_fe)
    X_test_scaled = scaler.transform(X_test_fe)
    
    print(f"После масштабирования: {X_train_scaled.shape}")
    
    # Feature Selection - отбор признаков по корреляции
    print("\n3. FEATURE SELECTION - ОТБОР ПО КОРРЕЛЯЦИИ")
    print("-" * 60)
    
    # Быстрый фильтр по корреляции с таргетом
    print("Отбор признаков по корреляции с таргетом...")
    corr_filter = CorrelationFilterTransformer(threshold=0.0)
    X_train_final = corr_filter.fit_transform(X_train_scaled, y_train)
    X_test_final = corr_filter.transform(X_test_scaled)
    print(f"  После фильтрации по корреляции: {X_train_final.shape[1]} признаков")
    
    # Дополнительная фильтрация: VarianceThreshold (удаляем признаки с нулевой дисперсией)
    print("  Фильтрация по дисперсии...")
    vt_transformer = VarianceThresholdTransformer(threshold=0.0)
    X_train_final = vt_transformer.fit_transform(X_train_final)
    X_test_final = vt_transformer.transform(X_test_final)
    print(f"  ✓ После фильтрации по дисперсии: {X_train_final.shape[1]} признаков")
    
    
    # ========================================================================
    # ШАГ 6: MODEL TRAINER (LightGBM с оптимизацией гиперпараметров)
    # ========================================================================
    print("\n" + "="*70)
    print("ШАГ 6: MODEL TRAINER - LIGHTGBM BASELINE")
    print("="*70)
    
    # Оптимизация гиперпараметров
    print("\n>>> ОПТИМИЗАЦИЯ ГИПЕРПАРАМЕТРОВ (Optuna)")
    print("=" * 70)
    
    print("  - 3 фолда при оптимизации: быстро оценить качество параметров")
    print("  - 5 фолдов при финальном обучении: стабильная оценка и лучшие предсказания")
    print("=" * 70)
    
    use_optuna = True  # 
    if use_optuna and OPTUNA_AVAILABLE:
        print("\n>>> Запуск оптимизации...")
        optuna_optimizer = OptunaOptimizer(model_type='lgb', n_trials=15, cv=3)
        best_lgb_params = optuna_optimizer.optimize(X_train_final, y_train)
        
        lgb_params = {
            'objective': 'binary',
            'metric': 'auc',
            'boosting_type': 'gbdt',
            'verbose': -1,
            'random_state': 42,
            'force_col_wise': True,
            **best_lgb_params
        }
        print(">>> Оптимизация завершена! Используются оптимизированные параметры.")
    else:
        # Хорошие базовые параметры для сильного baseline
        lgb_params = {
            'objective': 'binary',
            'metric': 'auc',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'min_data_in_leaf': 20,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
            'verbose': -1,
            'random_state': 42,
            'force_col_wise': True
        }
        if not OPTUNA_AVAILABLE:
            print("Используются базовые параметры LightGBM (Optuna недоступна)")
        else:
            print("Используются базовые параметры LightGBM (Optuna отключена)")
    
    # Обучение модели с быстрой CV стратегией (5 фолдов - стандарт)
    print("\n>>> ОБУЧЕНИЕ LIGHTGBM С КРОСС-ВАЛИДАЦИЕЙ")
    
    
    lgb_model = LightGBMClassifierCV(lgb_params=lgb_params, cv=5)  # Стандартные 5 фолдов
    lgb_model.fit(X_train_final, y_train)
    lgb_cv_score = lgb_model.cv_score_
    lgb_cv_std = np.std(lgb_model.cv_scores_)
    
    # ========================================================================
    # РЕЗУЛЬТАТЫ BASELINE
    # ========================================================================
    print("\n" + "="*70)
    print("РЕЗУЛЬТАТЫ BASELINE (LIGHTGBM)")
    print("="*70)
    print(f"Средний ROC-AUC: {lgb_cv_score:.5f} (+/- {lgb_cv_std:.5f})")
    print(f"ROC-AUC по фолдам:")
    for i, score in enumerate(lgb_model.cv_scores_, 1):
        print(f"  Fold {i}: {score:.5f}")
    print("="*70)
    
    # Предсказания на тестовой выборке
    print("\n>>> ГЕНЕРАЦИЯ ПРЕДСКАЗАНИЙ НА ТЕСТОВОЙ ВЫБОРКЕ")
    y_pred_proba = lgb_model.predict_proba(X_test_final)[:, 1]
    
    print(f"✓ Форма предсказаний: {y_pred_proba.shape}")
    print(f"✓ Диапазон: [{y_pred_proba.min():.4f}, {y_pred_proba.max():.4f}]")
    print(f"✓ Среднее значение: {y_pred_proba.mean():.4f}")
    
    # Создание submission файла
    submission = pd.DataFrame({
        'id': test_ids,
        'smoking': y_pred_proba
    })
    
    # Проверка корректности
    assert len(submission) == len(test_ids), "Количество строк не совпадает!"
    assert submission['smoking'].min() >= 0, "Есть отрицательные вероятности!"
    assert submission['smoking'].max() <= 1, "Есть вероятности больше 1!"
    assert submission['smoking'].isnull().sum() == 0, "Есть пропущенные значения!"
    
    # Сохранение submission файла
    submission_filename = f'submission_baseline_lgb_cv{np.round(lgb_cv_score, 4)}.csv'
    submission.to_csv(submission_filename, index=False)
    
    print(f"\n✓ Submission файл сохранен: {submission_filename}")
    print(f"✓ Размер файла: {submission.shape[0]} строк")
    
    # Информация о финальном DataFrame
    print("\n" + "="*70)
    print("ИНФОРМАЦИЯ О ФИНАЛЬНОМ DATAFRAME")
    print("="*70)
    print(f"Финальный размер признаков: {X_train_final.shape[1]}")
    print(f"Тип данных: {type(X_train_final)}")
    print(f"Форма: {X_train_final.shape}")
    print(f"Диапазон значений: [{X_train_final.min():.4f}, {X_train_final.max():.4f}]")
    print(f"Среднее значение: {X_train_final.mean():.4f}")
    print(f"Стандартное отклонение: {X_train_final.std():.4f}")
    print("="*70)
    
    print("\n" + "="*70)
    print("BASELINE ЗАВЕРШЕН!")
    print("="*70)
    print("Следующие шаги (опционально):")
    print("  - Добавить XGBoost и CatBoost")
    print("  - Создать ансамбль моделей")
    print("  - Применить Pseudo-labeling")
    print("="*70)


if __name__ == '__main__':
    main()
