from utility import dataset_function as reader
import pandas as pd
from sklearn.base import TransformerMixin, BaseEstimator
from sacred import Experiment
from typing import Callable, List


def empty_transform(X: pd.DataFrame) -> pd.DataFrame:
    return X


class BaseFeatureEngineering(TransformerMixin):
    dataset: pd.DataFrame
    function_transform: Callable
    keys: List

    def __init__(self, filename: str, keys: List, ex: Experiment, function_transform: Callable = empty_transform):
        self.dataset = reader.get_input(filename, ex)
        self.function_transform = function_transform
        self.keys = keys

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        assert isinstance(X, pd.DataFrame)
        Xt = X.merge(self.dataset, how="left", on=self.keys)
        Xt = self.function_transform(X=Xt)
        return Xt

