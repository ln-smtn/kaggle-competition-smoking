import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

# =============================
# Feature Engineering
# =============================

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

    test_ids = test['id']

    return X, y, X_test, test_ids

# =============================
# Main
# =============================

def main():
    X, y, X_test, test_ids = load_data()

    cat_cols = ['hearing(left)', 'hearing(right)', 'Urine protein', 'dental caries']
    num_cols = [c for c in X.columns if c not in cat_cols]

    # =============================
    # Preprocessing
    # =============================

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', 'passthrough', num_cols),
            ('cat', OneHotEncoder(
                categories=[
                    [0, 1],
                    [0, 1],
                    [1, 2, 3, 4, 5, 6],
                    [0, 1]
                ],
                handle_unknown='ignore',
                sparse_output=False,
                dtype=np.float32
            ), cat_cols)
        ]
    )

    # =============================
    # Fixed XGBoost parameters
    # =============================

    xgb_params = {'learning_rate': 0.018337449564255138, 'colsample_bytree': 0.2134019710295048, 'colsample_bylevel': 0.9036024360189447, 'subsample': 0.7424882202474626, 'reg_alpha': 0.017469961210395454, 'reg_lambda': 0.1315065611395556, 'max_depth': 12, 'min_child_weight': 46}
    xgb_params.update({'tree_method': 'hist',
                       'eval_metric': 'auc',
                       'n_estimators': 3000,
                       'n_jobs': -1,
                       'verbosity': 0})
    model = xgb.XGBClassifier(**xgb_params)

    # =============================
    # Pipeline
    # =============================

    pipeline = Pipeline([
        ('feat', ExtraFeatures()),
        ('prep', preprocessor),
        ('model', model)
    ])

    # =============================
    # Train on full data
    # =============================

    print("⏳ Training on full dataset...")
    pipeline.fit(X, y)
    print("✓ Training finished")


    # =============================
    # Predict test
    # =============================

    y_pred_proba = pipeline.predict_proba(X_test)[:, 1]

    # =============================
    # Submission
    # =============================

    submission = pd.DataFrame({
        'id': test_ids,
        'smoking': y_pred_proba
    })

    # Checks
    assert len(submission) == len(test_ids), "Количество строк не совпадает!"
    assert submission['smoking'].min() >= 0, "Есть отрицательные вероятности!"
    assert submission['smoking'].max() <= 1, "Есть вероятности больше 1!"
    assert submission['smoking'].isnull().sum() == 0, "Есть пропущенные значения!"

    submission_filename = 'submission_model_xg_1.csv'
    submission.to_csv(submission_filename, index=False)

    print(f"\n✓ Submission файл сохранен: {submission_filename}")
    print(f"✓ Размер файла: {submission.shape[0]} строк")

# =============================
# Run
# =============================

if __name__ == '__main__':
    main()
