import numpy as np
import pandas as pd
import xgboost as xgb
import optuna
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder


def print_trial_callback(study, trial):
    params = trial.params.copy()
    params.update({
        'n_estimators': 3000,
        'tree_method': 'hist',
        'eval_metric': 'auc'
    })
    print(
        f"\n[Trial {trial.number}] "
        f"AUC = {trial.value:.6f}\n"
        f"Params:\n{trial.params}\n"
        f"{'-'*50}"
    )

class ExtraFeatures(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        # Hearing
        best = np.minimum(X['hearing(left)'], X['hearing(right)'])
        worst = np.maximum(X['hearing(left)'], X['hearing(right)'])
        X['hearing(left)'] = best - 1
        X['hearing(right)'] = worst - 1

        # Eyesight
        X['eyesight(left)'] = np.where(X['eyesight(left)'] > 9, 0, X['eyesight(left)'])
        X['eyesight(right)'] = np.where(X['eyesight(right)'] > 9, 0, X['eyesight(right)'])
        best = np.minimum(X['eyesight(left)'], X['eyesight(right)'])
        worst = np.maximum(X['eyesight(left)'], X['eyesight(right)'])
        X['eyesight(left)'] = best
        X['eyesight(right)'] = worst

        # Clipping
        X['Gtp'] = np.clip(X['Gtp'], 0, 300)
        X['HDL'] = np.clip(X['HDL'], 0, 110)
        X['LDL'] = np.clip(X['LDL'], 0, 200)
        X['ALT'] = np.clip(X['ALT'], 0, 150)
        X['AST'] = np.clip(X['AST'], 0, 100)
        X['serum creatinine'] = np.clip(X['serum creatinine'], 0, 3)

        return X

# =============================
# Data Loader
# =============================

def load_data():
    train = pd.read_csv('playground-series-s3e24/train.csv')
    test = pd.read_csv('playground-series-s3e24/test.csv')

    y = train['smoking']
    X = train.drop(columns=['id', 'smoking'])
    X_test = test.drop(columns=['id'])

    return X, y, X_test

# =============================
# Optuna Objective
# =============================

class XGBOptunaOptimizer:
    def __init__(self, n_splits=10, n_trials=50, random_state=42):
        self.n_splits = n_splits
        self.n_trials = n_trials
        self.random_state = random_state

    def optimize(self, X, y, cat_cols):
        num_cols = [c for c in X.columns if c not in cat_cols]

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', RobustScaler(), num_cols),
                ('cat', OneHotEncoder(
                    categories=[[np.int64(0), np.int64(1)],
                                [np.int64(0), np.int64(1)],
                                [np.int64(1), np.int64(2), np.int64(3), np.int64(4), np.int64(5), np.int64(6)],
                                [np.int64(0), np.int64(1)]],
                    handle_unknown='ignore',
                    sparse_output=False,
                    dtype=np.float32
                ), cat_cols)
            ]
        )

        def objective(trial):
            params = {
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.03, log=True),
                'max_depth': trial.suggest_int('max_depth', 11, 13),
                'min_child_weight': trial.suggest_int('min_child_weight', 25, 60),
                'subsample': trial.suggest_float('subsample', 0.7, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.15, 0.25),
                'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.75, 0.95),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-6, 2e-1, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 2.0, log=True),
                'tree_method': 'hist',
                'eval_metric': 'auc',
                'n_estimators': 3000,
                'n_jobs': -1,
                'verbosity': 0,
            }

            skf = StratifiedKFold(
                n_splits=self.n_splits,
                shuffle=True,
                random_state=self.random_state
            )

            aucs = []
            for tr_idx, val_idx in skf.split(X, y):
                X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
                y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

                model = xgb.XGBClassifier(**params)

                pipe = Pipeline([
                    ('feat', ExtraFeatures()),
                    ('prep', preprocessor),
                    ('model', model)
                ])

                pipe.fit(
                    X_tr, y_tr,
                    model__eval_set=[(preprocessor.fit_transform(ExtraFeatures().fit_transform(X_val)), y_val)],
                    model__verbose=False
                )

                preds = pipe.predict_proba(X_val)[:, 1]
                aucs.append(roc_auc_score(y_val, preds))

            return float(np.mean(aucs))

        study = optuna.create_study(direction='maximize')
        study.optimize(objective,
                       n_trials=self.n_trials,
                       callbacks=[print_trial_callback],
                       show_progress_bar=True)
        return study

# =============================
# Main
# =============================

def main():
    X, y, X_test = load_data()

    cat_cols = ['hearing(left)', 'hearing(right)', 'Urine protein', 'dental caries']

    optimizer = XGBOptunaOptimizer(
        n_splits=10,
        n_trials=20
    )

    study = optimizer.optimize(X, y, cat_cols)

    print('Best AUC:', study.best_value)
    print('Best params:')
    for k, v in study.best_params.items():
        print(f'  {k}: {v}')


if __name__ == '__main__':
    main()