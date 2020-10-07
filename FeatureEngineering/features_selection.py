import pandas as pd
import numpy as np
from sklearn.base import TransformerMixin
import xgboost as xgb
import catboost as cb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
import shap
from typing import List


def calculate_permutation_importance(estimator, metric: callable,
                                     x_valid: pd.DataFrame,
                                     y_valid: pd.Series) -> pd.Series:
    """
    Расчет пермутированной важности признаков.
    """
    scores = {}

    y_pred = estimator.predict(xgb.DMatrix(x_valid))
    base_score = metric(y_valid, y_pred)

    for feature in tqdm(x_valid.columns):
        x_valid_copy = x_valid.copy()
        x_valid_copy[feature] = np.random.permutation(x_valid_copy[feature])

        y_pred = estimator.predict(xgb.DMatrix(x_valid_copy))
        score = metric(y_valid, y_pred)
        scores[feature] = base_score - score

    scores = pd.Series(scores)
    scores = scores.sort_values(ascending=False)

    return scores


class FeaturesSelector(TransformerMixin):
    column_count: int
    columns: List
    seed: int = 1

    def __init__(self, column_count=None, percent_column=None, seed=1):
        self.column_count = column_count
        self.percent_column = percent_column
        self.seed = seed

    def fit(self, X, y=None):
        assert isinstance(X, pd.DataFrame)
        print('-' * 20)
        print(f'All columns: {X.columns}')
        print('-' * 20)
        #self.columns = self.get_xgb_shap_importance(X, y)
        self.columns = self.get_catboost_shap_importance(X, y)
        return self

    def transform(self, X):
        assert isinstance(X, pd.DataFrame)
        return X[self.columns]

    def get_xgb_permutation_importance(self, X, y):
        xgb_params = {
            "booster": "gbtree",
            "objective": "binary:logistic",
            "eval_metric": "auc",
            "learning_rate": 0.1,
            "n_estimators": 1000,
            "nthread": 6,
            "seed": self.seed
        }
        Xt = X.copy()
        categorical_columns = Xt.select_dtypes(include=["object"]).columns
        Xt[categorical_columns] = Xt[categorical_columns].astype('str')
        Xt[categorical_columns] = Xt[categorical_columns].apply(LabelEncoder().fit_transform)

        x_train, x_valid, y_train, y_valid = train_test_split(Xt, y, train_size=0.7, random_state=self.seed,
                                                              shuffle=True, stratify=y)
        dtrain = xgb.DMatrix(x_train, y_train)
        dvalid = xgb.DMatrix(x_valid, y_valid)
        model = xgb.train(
            dtrain=dtrain,
            params=xgb_params,
            num_boost_round=500,
            evals=[(dtrain, "train"), (dvalid, "valid")],
            early_stopping_rounds=20,
            verbose_eval=10,
        )
        score = calculate_permutation_importance(model, roc_auc_score, x_valid, y_valid)

        if self.column_count is None and self.percent_column is None:
            return X.columns
        elif self.column_count is not None:
            count = min(self.column_count, len(Xt.columns))
            return list(score[:count].keys())
        else:
            count = int(len(Xt.columns) * self.percent_column)
            return list(score[:count].keys())

    def get_xgb_shap_importance(self, X, y):
        xgb_params = {
            "booster": "gbtree",
            "objective": "binary:logistic",
            "eval_metric": "auc",
            "learning_rate": 0.1,
            "n_estimators": 1000,
            "nthread": 6,
            "seed": self.seed
        }
        Xt = X.copy()
        categorical_columns = Xt.select_dtypes(include=["object"]).columns
        Xt[categorical_columns] = Xt[categorical_columns].astype('str')
        Xt[categorical_columns] = Xt[categorical_columns].apply(LabelEncoder().fit_transform)

        x_train, x_valid, y_train, y_valid = train_test_split(Xt, y, train_size=0.7, random_state=self.seed,
                                                              shuffle=True, stratify=y)
        model = xgb.XGBClassifier(**xgb_params)
        model.fit(X=x_train, y=y_train,
                  eval_set=[(x_train, y_train), (x_valid, y_valid)],
                  eval_metric='auc',
                  early_stopping_rounds=20,
                  verbose=True
                  )
        return self.shap_importance(model, x_valid)

    def get_catboost_shap_importance(self, X, y):
        catboost_params = {
            "n_estimators": 1000,
            "learning_rate": 0.1,
            "loss_function": "Logloss",
            "eval_metric": "AUC",
            "verbose": 10,
            "max_depth": 6,
            "early_stopping_rounds": 10,
            "random_seed": self.seed
        }
        Xt = X.copy()
        categorical_columns = Xt.select_dtypes(include=["object"]).columns
        Xt[categorical_columns] = Xt[categorical_columns].astype('str')

        x_train, x_valid, y_train, y_valid = train_test_split(Xt, y, train_size=0.7, random_state=self.seed,
                                                              shuffle=True, stratify=y)
        model = cb.CatBoostClassifier(**catboost_params)
        model.fit(X=x_train, y=y_train,
                  eval_set=[(x_train, y_train), (x_valid, y_valid)],
                  cat_features=categorical_columns,
                  verbose=True
                  )
        return self.shap_importance(model, x_valid)

    def shap_importance(self, model, x_valid):
        shap_values = shap.TreeExplainer(model).shap_values(x_valid)
        df_shape = pd.DataFrame(shap_values, columns=x_valid.columns)
        score = pd.Series(np.abs(df_shape).mean()).sort_values(ascending=False)
        print(f'get_shap_importance')
        pd_max_row = pd.get_option('max_rows')
        pd.set_option('max_rows', None)
        print(score)
        pd.set_option('max_rows', pd_max_row)
        if self.column_count is None and self.percent_column is None:
            count = len(score.values > 0)
            count = max(7, count)
            return list(score[:count].keys())
        elif self.column_count is not None:
            count = len(score > 0)
            count = min(self.column_count, count)
            return list(score[:count].keys())
        else:
            count = len(score > 0)
            count = max(7, int(count * self.percent_column))
            return list(score[:count].keys())
