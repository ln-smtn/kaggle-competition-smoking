"""
Стекинг на основе Best файлов
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# DATA COLLECTION & SPLIT
# ============================================================================

def get_input():
    """Загрузка и подготовка данных"""
    import os
    # Определяем путь к данным относительно текущего файла
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(os.path.dirname(base_dir), 'playground-series-s3e24')
    
    train = pd.read_csv(os.path.join(data_dir, 'train.csv'))
    test = pd.read_csv(os.path.join(data_dir, 'test.csv'))

    # Сохранение id для submission
    train_ids = train['id'].copy()
    test_ids = test['id'].copy()

    # Выделение целевой переменной
    y_train = train['smoking'].copy()

    # Удаление id и целевой переменной из признаков
    X_train = train.drop(['id', 'smoking'], axis=1)
    X_test = test.drop(['id'], axis=1)

    return X_train, y_train, X_test, test_ids


# ============================================================================
# FEATURE ENGINEERING (та же логика, что в Best файлах)
# ============================================================================

def create_extra_features(df):
    """Создание дополнительных признаков"""
    # order the ears
    best = np.where(df['hearing(left)'] < df['hearing(right)'],
                    df['hearing(left)'],  df['hearing(right)'])
    worst = np.where(df['hearing(left)'] < df['hearing(right)'],
                     df['hearing(right)'],  df['hearing(left)'])
    df['hearing(left)'] = best - 1
    df['hearing(right)'] = worst - 1

    # order the eyes - eyesight is worst to best, and 9+ should be worst!
    df['eyesight(left)'] = np.where(df['eyesight(left)'] > 9, 0, df['eyesight(left)'])
    df['eyesight(right)'] = np.where(df['eyesight(right)'] > 9, 0, df['eyesight(right)'])
    best = np.where(df['eyesight(left)'] < df['eyesight(right)'],
                    df['eyesight(left)'],  df['eyesight(right)'])
    worst = np.where(df['eyesight(left)'] < df['eyesight(right)'],
                     df['eyesight(right)'],  df['eyesight(left)'])
    df['eyesight(left)'] = best
    df['eyesight(right)'] = worst
    ##
    df['Gtp'] = np.clip(df['Gtp'], 0, 300)
    df['HDL'] = np.clip(df['HDL'], 0, 110)
    df['LDL'] = np.clip(df['LDL'], 0, 200)
    df['ALT'] = np.clip(df['ALT'], 0, 150)
    df['AST'] = np.clip(df['AST'], 0, 100)
    df['serum creatinine'] = np.clip(df['serum creatinine'], 0, 3)


def category_encoding(train_category, test_category):
    """One hot encoding категориальных признаков"""
    train_encode = pd.get_dummies(train_category, columns=['hearing(left)', 'hearing(right)', 'Urine protein', 'dental caries'])
    test_encode = pd.get_dummies(test_category, columns=['hearing(left)', 'hearing(right)', 'Urine protein', 'dental caries'])
    return train_encode, test_encode


class DataPreprocessor(BaseEstimator, TransformerMixin):
    """Трансформер для предобработки данных (для использования в Pipeline)"""
    
    def __init__(self, scaler=None, category_cols=None):
        self.scaler = scaler
        self.category_cols = category_cols
        self.fitted_scaler_ = None
        
    def fit(self, X, y=None):
        if self.scaler is None:
            self.fitted_scaler_ = RobustScaler()
        else:
            self.fitted_scaler_ = self.scaler
        
        # Применяем create_extra_features
        X_copy = X.copy()
        create_extra_features(X_copy)
        
        # Разделение на категориальные и числовые
        train_to_scale = X_copy.drop(self.category_cols, axis=1)
        self.fitted_scaler_.fit(train_to_scale)
        
        return self
    
    def transform(self, X):
        X_copy = X.copy()
        create_extra_features(X_copy)
        
        # Разделение на категориальные и числовые
        X_to_scale = X_copy.drop(self.category_cols, axis=1)
        X_category = X_copy[self.category_cols]
        
        # Стандартизация числовых признаков
        scaled_X = pd.DataFrame(
            self.fitted_scaler_.transform(X_to_scale),
            columns=X_to_scale.columns
        )
        
        # Кодирование категориальных признаков
        # Для transform нужно использовать тот же подход
        X_encode = pd.get_dummies(X_category, columns=['hearing(left)', 'hearing(right)', 'Urine protein', 'dental caries'])
        
        # Объединение
        scaled_X = scaled_X.reset_index(drop=True)
        X_encode = X_encode.reset_index(drop=True)
        
        result = pd.concat([X_encode, scaled_X], axis=1)
        return result


def data_preprocessing(X_train, y_train, X_test, scaler=None, category_cols=None, do_category_encoding=True):
    """Предобработка данных (та же логика, что в Best файлах)"""
    create_extra_features(X_train)
    create_extra_features(X_test)

    # Разделение данных на категориальные и числовые признаки
    train_to_scale = X_train.drop(category_cols, axis=1)
    train_category = X_train[category_cols]
    test_to_scale = X_test.drop(category_cols, axis=1)
    test_category = X_test[category_cols]

    if scaler is None:
        scaler = RobustScaler()

    # Стандартизация данных (только числовые признаки)
    scaled_train = pd.DataFrame(scaler.fit_transform(train_to_scale), columns=train_to_scale.columns)
    scaled_test = pd.DataFrame(scaler.transform(test_to_scale), columns=test_to_scale.columns)

    # Кодирование категориальных признаков
    train_encode, test_encode = category_encoding(train_category, test_category)

    # Объединение кодированных категориальных и стандартизованных числовых признаков
    train_encode = train_encode.reset_index(drop=True)
    test_encode = test_encode.reset_index(drop=True)
    scaled_train = scaled_train.reset_index(drop=True)
    scaled_test = scaled_test.reset_index(drop=True)

    train_df = pd.concat([train_encode, scaled_train], axis=1)
    test_df = pd.concat([test_encode, scaled_test], axis=1)

    return train_df, test_df


# ============================================================================
# ЛУЧШИЕ ПАРАМЕТРЫ ИЗ BEST ФАЙЛОВ
# ============================================================================

# Из Best_LGB.py
LGB_PARAMS = {
    'learning_rate': 0.01184431975182039,
    'num_leaves': 245,
    'max_depth': 10,
    'min_child_samples': 32,
    'subsample': 0.6624074561769746,
    'colsample_bytree': 0.662397808134481,
    'reg_alpha': 2.5502648504032812e-08,
    'reg_lambda': 0.011567327199145964,
    'n_estimators': 2083,
    'objective': 'binary',
    'metric': 'auc',
    'n_jobs': -1,
    'verbosity': -1
}

# Из Best_XG_boost.py (актуальные параметры)
XGB_PARAMS = {
    'learning_rate': 0.017940848436017145,
    'max_depth': 11,
    'min_child_weight': 60,
    'subsample': 0.9542993050541952,
    'colsample_bytree': 0.21497203607822757,
    'colsample_bylevel': 0.8724464985284567,
    'reg_alpha': 0.002852523609332756,
    'reg_lambda': 0.1462651585929734,
    'tree_method': 'hist',
    'eval_metric': 'auc',
    'n_estimators': 3000,
    'n_jobs': -1,
    'verbosity': 0,
    'random_state': 42
}

# Из Best_CAT.py
CAT_PARAMS = {
    'learning_rate': 0.04056956101904861,
    'depth': 7,
    'l2_leaf_reg': 7.459199917293563,
    'border_count': 230,
    'bagging_temperature': 0.44856780106647864,
    'random_strength': 2.333989054467297,
    'subsample': 0.7742421038427931,
    'colsample_bylevel': 0.7364181387936571,
    'iterations': 2390,
    'min_data_in_leaf': 11,
    'loss_function': 'Logloss',
    'eval_metric': 'AUC',
    'random_state': 42,
    'thread_count': -1,
    'verbose': False,
    'allow_writing_files': False
}


# ============================================================================
# СТЕКИНГ С МЕТА-МОДЕЛЬЮ
# ============================================================================

def create_meta_features(X_train, y_train, models_config, cv=5):
    """
    Создает мета-признаки для стекинга (out-of-fold predictions)
    
    """
    
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    n_samples = len(X_train)
    n_models = len(models_config)
    meta_features = np.zeros((n_samples, n_models))
    

    print(f"Количество фолдов (K): {cv}")
    print(f"Количество моделей: {n_models}")
    print(f"\nПроцесс:")
    print(f"  Для каждой модели:")
    print(f"    - Проходим по {cv} фолдам")
    print(f"    - На каждом фолде: обучаем на train → предсказания на val")
    print(f"    - Получаем out-of-fold предсказания для всех образцов")
    
    # ========================================================================
    # Для каждой модели i получаем out-of-fold предсказания
    # ========================================================================
    for model_idx, model_config in enumerate(models_config):
        model_name = model_config['name']
        model_class = model_config['class']
        model_params = model_config['params']
        
        print(f"\n{'─'*70}")
        print(f"Модель {model_idx+1}/{n_models}: {model_name.upper()}")
        print(f"{'─'*70}")
        
        # Вектор для хранения предсказаний модели i для всех образцов
        model_predictions = np.zeros(n_samples)
        
        # ====================================================================
        # K-Fold схема: получаем вероятности для каждого фолда
        # ====================================================================
        for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train), 1):
            X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
            
            # ШАГ 1: Обучаем модель i на train части фолда k
            model = model_class(**model_params)
            
            # Специальная обработка для CatBoost
            if isinstance(model, CatBoostClassifier):
                model.fit(X_tr, y_tr, verbose=False)
            else:
                model.fit(X_tr, y_tr)
            
            # ШАГ 2: Получаем вероятности на validation части фолда k
            if hasattr(model, 'predict_proba'):
                pred = model.predict_proba(X_val)[:, 1]  # Вероятность положительного класса
            else:
                pred = model.predict(X_val)
            
            # ШАГ 3: Сохраняем предсказания в соответствующие позиции
            # model_predictions[val_idx] содержит предсказания для образцов из validation фолда
            model_predictions[val_idx] = pred
            
            # Вычисляем AUC на этом фолде для контроля качества
            fold_auc = roc_auc_score(y_val, pred)
            print(f"  Fold {fold}/{cv}: Train={len(train_idx)}, Val={len(val_idx)}, AUC={fold_auc:.6f}")
        
        # Сохраняем предсказания модели i в столбец i матрицы мета-признаков
        meta_features[:, model_idx] = model_predictions
        
        # Итоговая статистика по модели i
        overall_auc = roc_auc_score(y_train, model_predictions)
        print(f"\n  ✓ {model_name.upper()} завершена")
        print(f"    OOF AUC (out-of-fold): {overall_auc:.6f}")
        print(f"    Диапазон предсказаний: [{model_predictions.min():.4f}, {model_predictions.max():.4f}]")
        print(f"    Среднее значение: {model_predictions.mean():.4f}")
    
    
    print(f"Форма матрицы мета-признаков: {meta_features.shape}")
    print(f"  - Строки: {meta_features.shape[0]} образцов")
    print(f"  - Столбцы: {meta_features.shape[1]} моделей")
    print(f"  - meta_features[i, j] = предсказание модели j для образца i")
    
    # Вычисляем и сохраняем метрики для каждой модели
    model_metrics = {}
    for model_idx, model_config in enumerate(models_config):
        model_name = model_config['name']
        model_predictions = meta_features[:, model_idx]
        model_auc = roc_auc_score(y_train, model_predictions)
        model_metrics[model_name] = model_auc
    
    return meta_features, model_metrics


def train_stacking_ensemble(X_train, y_train, X_test, models_config, meta_model=None, cv=5):
    """
    Обучает стекинг ансамбль
    
    Args:
        X_train: обучающие данные (уже обработанные)
        y_train: целевая переменная
        X_test: тестовые данные (уже обработанные)
        models_config: список конфигураций базовых моделей
        meta_model: мета-модель (по умолчанию LogisticRegression)
        cv: количество фолдов для создания мета-признаков
    
    Returns:
        ensemble_pred: предсказания на тестовых данных
        fitted_base_models: список обученных базовых моделей
        fitted_meta_model: обученная мета-модель
        meta_features_train: мета-признаки на train данных
        base_models_metrics: словарь с метриками базовых моделей {'lgb': auc, 'xgb': auc, ...}
        meta_model_auc: AUC мета-модели на train данных
    """
    
    if meta_model is None:
        meta_model = LogisticRegression(random_state=42, max_iter=1000)
    
    # 1. Создаем мета-признаки на train данных (out-of-fold)
    meta_features_train, base_models_metrics = create_meta_features(X_train, y_train, models_config, cv=cv)
    
    # 2. Обучаем мета-модель на мета-признаках
    
    meta_model.fit(meta_features_train, y_train)
    meta_model_auc = roc_auc_score(y_train, meta_model.predict_proba(meta_features_train)[:, 1])
    
    print(f"\n✓ Мета-модель обучена")
    print(f"✓ AUC мета-модели на train (мета-признаках): {meta_model_auc:.6f}")
    print(f"\nМета-модель научилась комбинировать предсказания базовых моделей!")
    
    # 3. Обучаем базовые модели на всех train данных
    print(f"{'='*70}")
    fitted_base_models = []
    
    for model_config in models_config:
        model_name = model_config['name']
        model_class = model_config['class']
        model_params = model_config['params']
        
        print(f"Обучение {model_name.upper()}...")
        model = model_class(**model_params)
        
        if isinstance(model, CatBoostClassifier):
            model.fit(X_train, y_train, verbose=False)
        else:
            model.fit(X_train, y_train)
        
        fitted_base_models.append(model)
        print(f"  ✓ {model_name.upper()} готов")
    
    # 4. Создаем мета-признаки на test данных
    
    meta_features_test = np.zeros((len(X_test), len(fitted_base_models)))
    
    for model_idx, model in enumerate(fitted_base_models):
        model_name = models_config[model_idx]['name']
        if hasattr(model, 'predict_proba'):
            pred = model.predict_proba(X_test)[:, 1]
        else:
            pred = model.predict(X_test)
        meta_features_test[:, model_idx] = pred
        print(f"  ✓ {model_name.upper()}: диапазон [{pred.min():.4f}, {pred.max():.4f}]")
    
    # 5. Предсказания мета-модели
    
    ensemble_pred = meta_model.predict_proba(meta_features_test)[:, 1]
    print(f"✓ Финальные предсказания готовы")
    print(f"  Диапазон: [{ensemble_pred.min():.4f}, {ensemble_pred.max():.4f}]")
    print(f"  Среднее значение: {ensemble_pred.mean():.4f}")
    
    return ensemble_pred, fitted_base_models, meta_model, meta_features_train, base_models_metrics, meta_model_auc

def main():
    
    # ========================================================================
    # ШАГ 1: DATA COLLECTION & SPLIT
    # ========================================================================
    
    X_train, y_train, X_test, test_ids = get_input()
    
    print(f"✓ Размер обучающей выборки: {X_train.shape}")
    print(f"✓ Размер тестовой выборки: {X_test.shape}")
    print(f"✓ Размер целевой переменной: {y_train.shape}")
    print(f"✓ Распределение классов: {np.bincount(y_train)}")
    
    # ========================================================================
    # ШАГ 2: DATA PREPROCESSING
    # ========================================================================
    
    category_cols = ['hearing(left)', 'hearing(right)', 'Urine protein', 'dental caries']
    
    X_train_processed, X_test_processed = data_preprocessing(
        X_train, y_train, X_test,
        scaler=None,  # Используем RobustScaler по умолчанию
        category_cols=category_cols,
        do_category_encoding=True
    )
    
    print(f"✓ После Feature Engineering: {X_train_processed.shape[1]} признаков")
    print(f"✓ Форма train: {X_train_processed.shape}")
    print(f"✓ Форма test: {X_test_processed.shape}")
    
    # ========================================================================
    # ШАГ 3: КОНФИГУРАЦИЯ БАЗОВЫХ МОДЕЛЕЙ
    # ========================================================================
   
    
    models_config = [
        {
            'name': 'lgb',
            'class': lgb.LGBMClassifier,
            'params': LGB_PARAMS
        },
        {
            'name': 'xgb',
            'class': xgb.XGBClassifier,
            'params': XGB_PARAMS
        },
        {
            'name': 'cat',
            'class': CatBoostClassifier,
            'params': CAT_PARAMS
        }
    ]
    
    print(f"✓ Настроено моделей: {len(models_config)}")
    for model_config in models_config:
        print(f"  - {model_config['name'].upper()}")
    
    # ========================================================================
    # ШАГ 4: ОБУЧЕНИЕ СТЕКИНГ АНСАМБЛЯ
    # ========================================================================
    ensemble_pred, fitted_base_models, meta_model, meta_features_train, base_models_metrics, meta_model_auc = train_stacking_ensemble(
        X_train_processed, y_train, X_test_processed,
        models_config,
        meta_model=LogisticRegression(random_state=42, max_iter=1000),
        cv=5
    )
    
    # ========================================================================
    # ШАГ 5: СОХРАНЕНИЕ SUBMISSION
    # ========================================================================
    
    
    submission = pd.DataFrame({
        'id': test_ids,
        'smoking': ensemble_pred
    })
    
    # Проверка корректности
    assert len(submission) == len(test_ids), "Количество строк не совпадает!"
    assert submission['smoking'].min() >= 0, "Есть отрицательные вероятности!"
    assert submission['smoking'].max() <= 1, "Есть вероятности больше 1!"
    assert submission['smoking'].isnull().sum() == 0, "Есть пропущенные значения!"
    
    # Сохранение submission файла
    submission_filename = 'submission_stacking.csv'
    submission.to_csv(submission_filename, index=False)
    
    print(f"✓ Submission файл сохранен: {submission_filename}")
    print(f"✓ Размер файла: {submission.shape[0]} строк")
    
    # ========================================================================
    # ИТОГОВАЯ ИНФОРМАЦИЯ И МЕТРИКИ
    # ========================================================================
    
    print("\n📊 МЕТРИКИ БАЗОВЫХ МОДЕЛЕЙ (OOF - out-of-fold):")
    print("-" * 70)
    for model_name, auc in sorted(base_models_metrics.items(), key=lambda x: x[1], reverse=True):
        print(f"  {model_name.upper():8s}: {auc:.6f}")
    
    print(f"\n📊 МЕТРИКА МЕТА-МОДЕЛИ ({type(meta_model).__name__}):")
    print("-" * 70)
    print(f"  Мета-модель: {meta_model_auc:.6f}")
    
    # Вычисляем улучшение
    best_base_auc = max(base_models_metrics.values())
    improvement = meta_model_auc - best_base_auc
    improvement_pct = (improvement / best_base_auc) * 100
    
    print(f"\n📈 СРАВНЕНИЕ:")
    print("-" * 70)
    print(f"  Лучшая базовая модель: {max(base_models_metrics.items(), key=lambda x: x[1])[0].upper()}")
    print(f"  AUC лучшей базовой модели: {best_base_auc:.6f}")
    print(f"  AUC мета-модели:         {meta_model_auc:.6f}")
    print(f"  Улучшение:               {improvement:+.6f} ({improvement_pct:+.3f}%)")
    
    if improvement > 0:
        print(f"  ✓ Мета-модель улучшила результат!")
    else:
        print(f"  ⚠ Мета-модель не улучшила результат (возможно, нужна настройка)")
    
    
    print(f"Использовано базовых моделей: {len(fitted_base_models)}")
    print(f"Мета-модель: {type(meta_model).__name__}")
    print(f"Submission файл: {submission_filename}")
    print(f"Диапазон предсказаний: [{ensemble_pred.min():.4f}, {ensemble_pred.max():.4f}]")
    print(f"Среднее значение: {ensemble_pred.mean():.4f}")
    print("\n" + "="*70)
    print("ИТОГОВАЯ МЕТРИКА СТЕКИНГА:")
    print(f"  ROC-AUC: {meta_model_auc:.6f}")
    print("="*70)


if __name__ == '__main__':
    main()

