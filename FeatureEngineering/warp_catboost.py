import catboost as cb
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sacred import Experiment
import time
from sklearn.model_selection import KFold


class WarpCatboost(BaseEstimator, TransformerMixin):
    cat_features: list
    model: cb.CatBoostClassifier
    seed: int
    ex: Experiment
    cv: KFold

    def __init__(self, params, KFold_split: int = True, ex: Experiment = None):
        self.model = cb.CatBoostClassifier(**params)
        self.seed = params['random_seed']
        self.numerical = []
        self.cat_features = []
        self.ex = ex
        self.KFold_split = KFold_split
        self.estimators = None

    def fit(self, X, y=None):

        X.info()

        if self.KFold_split > 1:
            self.cv = KFold(n_splits=self.KFold_split, random_state=1234123, shuffle=True)
            return self.catboost_cross_validation(self.transform(X), y)
        else:
            return self.catboost_hold_out_validation(self.transform(X), y)

    def transform(self, X):
        assert isinstance(X, pd.DataFrame)
        self.cat_features = X.dtypes[X.dtypes == "object"].index
        self.numerical = list(set(X.columns) - set(self.cat_features))
        X[self.cat_features] = X[self.cat_features].astype(str)
        X[self.numerical] = X[self.numerical].astype(float)
        return X

    def predict(self, X):
        Xt = self.transform(X.copy())
        if self.KFold_split > 1:
            y_pred = np.zeros(Xt.shape[0])
            for estimator in self.estimators:
                y_pred += estimator.predict_proba(Xt)[:, 1]
            return np.round(y_pred / len(self.estimators))
        else:
            return self.model.predict(Xt)

    def predict_proba(self, X):
        Xt = self.transform(X.copy())
        if self.KFold_split > 1:
            y_pred = np.zeros(Xt.shape[0])
            for estimator in self.estimators:
                y_pred += estimator.predict_proba(Xt)[:, 1]
            return y_pred / len(self.estimators)
        else:
            return self.model.predict_proba(Xt)[:, 1]

    def logging_metric(self, name_metric: str, value: float, step=0):
        if self.ex is not None:
            self.ex.log_scalar(name_metric, value, step)

    def catboost_hold_out_validation(self, X, y, split_params=[0.7, 0.2, 0.1]):
        """
        Hold-Out валидация для модели catbooost.

        Parameters
        ----------
       X: pandas.core.frame.DataFrame
            Матрица признако для обучения модели.

        y: pandas.core.frame.Series
            Вектор целевой переменной для обучения модели.

        split_params: List[float], optional, default = [0.7, 0.2, 0.1]
            Параметры (доли) разбиения выборки.
            Опциональный параметр, по умолчанию, равен [0.7, 0.2, 0.1].

        categorical: str, optional, default = None
            Список категориальных признаков.
            Опциональный параметр, по умолчанию, не используется.

        Returns
        -------
        estimator: catboost.core.CatBoostClassifier
            Обученный классификатор catboost.
        """
        x_train, x_valid = train_test_split(
            X, train_size=split_params[0], random_state=self.seed
        )
        y_train, y_valid = train_test_split(
            y, train_size=split_params[0], random_state=self.seed
        )

        if len(split_params) == 3:
            test_size = int(split_params[2] * X.shape[0])

            x_valid, x_test = train_test_split(
                x_valid, test_size=test_size, random_state=self.seed
            )
            y_valid, y_test = train_test_split(
                y_valid, test_size=test_size, random_state=self.seed
            )

        self.model.fit(X=x_train, y=y_train,
                       cat_features=self.cat_features,
                       eval_set=[(x_train, y_train), (x_valid, y_valid)]
                       )

        print("= " * 80)
        valid_score = roc_auc_score(y_valid, self.model.predict_proba(x_valid)[:, 1])
        print(f"Valid Score = {round(valid_score, 4)}")
        self.logging_metric('Valid Score', round(valid_score, 4))
        if len(split_params) == 3:
            test_prediction = self.model.predict_proba(x_test)[:, 1]
            test_score = roc_auc_score(y_test, test_prediction)
            print(f"Test Score = {round(test_score, 4)}")
            self.logging_metric('Test Score', round(test_score, 4))
        return self

    def catboost_cross_validation(self, X, y):
        """
        Кросс-валидация для модели catbooost.

        Parameters
        ----------
        X: pandas.core.frame.DataFrame
            Матрица признако для обучения модели.

        y: pandas.core.frame.Series
            Вектор целевой переменной для обучения модели.


        Returns
        -------
        estimators: list
            Список с объектами обученной модели.


        """
        estimators, folds_scores = [], []
        oof_preds = np.zeros(X.shape[0])

        print(f"{time.ctime()}, Cross-Validation, {X.shape[0]} rows, {X.shape[1]} cols")

        for fold, (train_idx, valid_idx) in enumerate(self.cv.split(X, y)):
            x_train, x_valid = X.loc[train_idx], X.loc[valid_idx]
            y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

            estimator = self.model
            estimator.fit(
                x_train, y_train, self.cat_features,
                eval_set=[(x_train, y_train), (x_valid, y_valid)]
            )
            oof_preds[valid_idx] = estimator.predict_proba(x_valid)[:, 1]
            score = roc_auc_score(y_valid, oof_preds[valid_idx])
            print(f"Fold {fold + 1}, Valid score = {round(score, 5)}")
            self.logging_metric(f"Fold Valid score", round(score, 5), fold + 1)
            folds_scores.append(round(score, 5))
            estimators.append(estimator)

        print(f"Score by each fold: {folds_scores}")
        print("= " * 65)
        self.logging_metric("OOF-score", roc_auc_score(y, oof_preds))
        self.estimators = estimators
        return self
