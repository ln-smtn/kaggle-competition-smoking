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

import optuna
from boruta import BorutaPy


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

    return train_encode, test_encode

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

class OptimizerLGB:
    def __init__(
            self,
            n_splits: int = 5,
            n_trials: int = 50,
            random_state: int = 42
    ):
        self.n_splits = n_splits
        self.n_trials = n_trials
        self.random_state = random_state

        self.best_auc = -np.inf
        self.best_params = None
        self.best_model = None

        self.skf = StratifiedKFold(
            n_splits=self.n_splits,
            shuffle=True,
            random_state=self.random_state
        )

    # --------------------------------------------------
    # Public API
    # --------------------------------------------------
    def optimize(self, X, y):

        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=self.random_state)
        )

        study.optimize(
            lambda trial: self._objective(trial, X, y),
            n_trials=self.n_trials,
            show_progress_bar=True
        )

        print("\n===== OPTIMIZATION FINISHED =====")
        print(f"Best CV AUC: {study.best_value:.6f}")
        print("Best parameters:")
        for k, v in study.best_params.items():
            print(f"  {k}: {v}")

        return study

    # --------------------------------------------------
    # Objective
    # --------------------------------------------------
    def _objective(self, trial, X, y):

        params = self._suggest_params(trial)
        auc_scores = []

        for fold, (train_idx, valid_idx) in enumerate(self.skf.split(X, y), 1):
            X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
            y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

            model = lgb.LGBMClassifier(**params)
            model.fit(X_train, y_train)

            y_pred = model.predict_proba(X_valid)[:, 1]
            auc = roc_auc_score(y_valid, y_pred)
            auc_scores.append(auc)

        mean_auc = float(np.mean(auc_scores))

        print(
            f"[Trial {trial.number:03d}] "
            f"Mean AUC: {mean_auc:.6f} | "
            f"Best AUC: {self.best_auc:.6f}"
        )

        if mean_auc > self.best_auc:
            self.best_auc = mean_auc
            self.best_params = params

            self.best_model = lgb.LGBMClassifier(**params)
            self.best_model.fit(X, y)

        return mean_auc

    # --------------------------------------------------
    # Params
    # --------------------------------------------------
    @staticmethod
    def _suggest_params(trial):

        return {
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.05, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 16, 256),
            "max_depth": trial.suggest_int("max_depth", -1, 14),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 0.1, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 0.1, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 700, 3000),
            "objective": "binary",
            "metric": "auc",
            "n_jobs": -1,
            "verbosity": -1
        }


def main():

    print("ШАГ 1: DATA COLLECTION & SPLIT")

    X_train, y_train, X_test, test_ids = get_input()


    print(f"✓ Распределение классов: {np.bincount(y_train)}")


    category_cols = ['hearing(left)', 'hearing(right)', 'Urine protein','dental caries']
    X, test = data_preprocessing(
        X_train, y_train, X_test, scaler=None,
        category_cols=category_cols, do_category_encoding=True
    )

    print(f"✓ Размер обучающей выборки: {X.shape}")
    print(f"✓ Размер тестовой выборки: {test.shape}")
    print(f"✓ Размер целевой переменной: {y_train.shape}")

    optimizer = OptimizerLGB(
        n_splits=5,
        n_trials=20
    )

    study = optimizer.optimize(X, y_train)

    best_model = optimizer.best_model
    best_params = optimizer.best_params
    best_auc = optimizer.best_auc

    print(best_params)



if __name__ == '__main__':
    main()