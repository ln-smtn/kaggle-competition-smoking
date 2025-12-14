import numpy as np
import pandas as pd
import lightgbm as lgb
from tqdm import tqdm
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection._split import check_cv
from sklearn.base import clone, is_classifier
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score
from sklearn.feature_selection import (
    VarianceThreshold, SelectKBest, mutual_info_classif,
    SelectFromModel, RFE
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier, ElasticNet
import warnings
warnings.filterwarnings('ignore')


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
    import os
    # Определяем путь к данным относительно текущего файла
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

    return X_train, y_train, X_test, test_ids


def create_extra_features(df):
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

    # One hot encoding
    train_encode = pd.get_dummies(train_category, columns = ['hearing(left)', 'hearing(right)', 'Urine protein', 'dental caries'])
    test_encode  = pd.get_dummies(test_category, columns = ['hearing(left)', 'hearing(right)', 'Urine protein', 'dental caries'])

    # Сглаживание
    #for i in train_encode.columns :
    #    train_encode[i] = train_encode[i].apply(lambda x :0.9 if x ==1 else 0.1)

    return train_encode, test_encode


def data_preprocessing(X_train, y_train, X_test, scaler = None, category_cols = None, do_category_encoding=True):

    create_extra_features(X_train)
    create_extra_features(X_test)

    # Разделение данных на категориальные и числовые признаки
    train_to_scale = X_train.drop(category_cols,axis =1)
    train_category = X_train[category_cols]
    test_to_scale = X_test.drop(category_cols,axis =1)
    test_category = X_test[category_cols]

    if scaler is None:
        scaler = RobustScaler()

    # Стандартизация данных (только числовые признаки)
    scaled_train = pd.DataFrame(scaler.fit_transform(train_to_scale),columns = train_to_scale.columns)
    scaled_test = pd.DataFrame(scaler.transform(test_to_scale),columns = test_to_scale.columns)

    # Кодирование категориальных признаков
    train_encode, test_encode = category_encoding(train_category, test_category)

    # Объединение кодированных категориальных и стандартизованных числовых признаков
    train_encode = train_encode.reset_index(drop = True)
    test_encode = test_encode.reset_index(drop = True)
    scaled_train  = scaled_train.reset_index(drop =True)
    scaled_test  = scaled_test.reset_index(drop =True)

    train_df = pd.concat([train_encode,scaled_train],axis =1)
    test_df = pd.concat([test_encode,scaled_test],axis =1)

    train_df = create_new_features(train_df)
    test_df = create_new_features(test_df)

    return train_df, test_df

def create_new_features(df):
    df = df.copy()

    # BMI (Body Mass Index)
    if 'height(cm)' in df.columns and 'weight(kg)' in df.columns:
        df['BMI'] = df['weight(kg)'] / ((df['height(cm)'] / 100) ** 2)

    # Отношение талии к росту
    if 'waist(cm)' in df.columns and 'height(cm)' in df.columns:
        df['WHR'] = df['waist(cm)'] / df['height(cm)']

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

    # Заполнение NaN значений медианой
    df = df.fillna(df.median())


    return df

def main():

    print("ШАГ 1: DATA COLLECTION & SPLIT")

    X_train, y_train, X_test, test_ids = get_input()

    print(f"✓ Размер обучающей выборки: {X_train.shape}")
    print(f"✓ Размер тестовой выборки: {X_test.shape}")
    print(f"✓ Размер целевой переменной: {y_train.shape}")
    print(f"✓ Распределение классов: {np.bincount(y_train)}")


    category_cols = ['hearing(left)', 'hearing(right)', 'Urine protein','dental caries']

    sc = RobustScaler()

    n_splits = 5

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    auc_scores = []
    best_model = None
    best_auc = 0.0
    X, test = data_preprocessing(
        X_train, y_train, X_test, scaler=None,
        category_cols=category_cols, do_category_encoding=True
    )

    print("ИНФОРМАЦИЯ О ФИНАЛЬНОМ DATAFRAME")
    print(f"Финальный размер признаков: {X.shape[1]}")
    print(f"Форма: {X.shape}")
    y = y_train

    params = {'learning_rate': 0.014441453107851188, 'num_leaves': 137, 'max_depth': 9, 'min_child_samples': 34, 'subsample': 0.991604402309641, 'colsample_bytree': 0.6929374849647897, 'reg_alpha': 3.555255370903241e-05, 'reg_lambda': 8.454317934951585e-06, 'n_estimators': 2293, 'objective': 'binary', 'metric': 'auc', 'n_jobs': -1, 'verbosity': -1}


    model = lgb.LGBMClassifier(**params)

    model.fit(X_train, y_train)

    print("ГЕНЕРАЦИЯ ПРЕДСКАЗАНИЙ НА ТЕСТОВОЙ ВЫБОРКЕ")

    # Предсказания на тестовой выборке (как в примере)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

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
    submission_filename = f'submission_model_lgb.csv'
    submission.to_csv(submission_filename, index=False)

    print(f"\n✓ Submission файл сохранен: {submission_filename}")
    print(f"✓ Размер файла: {submission.shape[0]} строк")


if __name__ == '__main__':
    main()