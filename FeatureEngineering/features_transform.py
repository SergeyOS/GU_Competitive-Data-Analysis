import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.base import TransformerMixin, BaseEstimator
from functools import reduce
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline


class FeaturesTransform(TransformerMixin, BaseEstimator):
    base_pipeline = None
    numerical_features = []
    categorical_features = []

    def __init__(self, pca_component=10):
        self.base_pipeline = None
        self.categorical_features = []
        self.numerical_features = []
        self.pca_component = pca_component

    def fit(self, X, y=None):
        assert isinstance(X, pd.DataFrame)
        Xt = self.prepare(X)
        self.find_category_features(Xt)
        self.find_numeric_features(Xt)
        if self.pca_component > 0:
            PCA_line = Pipeline(steps=[
                ('extract_num', ColumnExtractor(self.numerical_features)),
                ('PCA', PCATransformer(self.pca_component))])
            numeric_and_PCA = DFFeatureUnion(
                transformer_list=[
                    ('num', ColumnExtractor(self.numerical_features), self.numerical_features)
                    , ('num_PCA', PCA_line, self.numerical_features)
                ])
        else:
            numeric_and_PCA = Pipeline(steps=[('num', ColumnExtractor(self.numerical_features)),
                                              ('generator', GeneratorNumFeatures(self.numerical_features))])

        self.base_pipeline = DFFeatureUnion(
            transformer_list=[
                ('cat', DFImputer(strategy='constant', fill_value='Missing'), self.categorical_features),
                ('numeric_line', numeric_and_PCA, self.numerical_features)
            ])
        self.base_pipeline.fit(Xt, y)
        return self

    def prepare(self, X):
        Xt = X.drop('application_number', axis=1)
        Xt = Xt.replace(np.inf, np.nan)
        Xt = Xt.replace(-np.inf, np.nan)
        return Xt

    def transform(self, X):
        assert isinstance(X, pd.DataFrame)
        Xt = self.prepare(X)
        Xt = self.base_pipeline.transform(Xt)
        return Xt

    def find_category_features(self, X):
        self.categorical_features = []
        self.categorical_features = list(X.select_dtypes(include=[np.object]).columns)
        for column in X.select_dtypes(exclude=[np.object]).columns:
            if X[column].nunique() < 10:
                self.categorical_features.append(column)
                X[column] = X[column].astype(np.object)

    def find_numeric_features(self, X):
        if len(self.categorical_features) == 0:
            self.find_category_features(X)
        self.numerical_features = []
        self.numerical_features = list(X.select_dtypes(exclude=[np.object]).columns)


class DFFeatureUnion(BaseEstimator, TransformerMixin):
    # FeatureUnion but for pandas DataFrames

    def __init__(self, transformer_list):
        self.transformer_list = transformer_list

    def fit(self, X, y=None):
        for (name, t, columns) in self.transformer_list:
            t.fit(X[columns], y)
        return self

    def transform(self, X):
        # assumes X is a DataFrame
        Xts = [t.transform(X[columns]) for _, t, columns in self.transformer_list]
        Xunion = reduce(lambda X1, X2: pd.merge(X1, X2, left_index=True, right_index=True), Xts)
        return Xunion


class DFImputer(BaseEstimator, TransformerMixin):
    # Imputer but for pandas DataFrames

    def __init__(self, strategy='mean', fill_value=np.nan):
        self.strategy = strategy
        self.imp = None
        self.statistics_ = None
        self.fill_value = fill_value

    def fit(self, X, y=None):
        self.imp = SimpleImputer(strategy=self.strategy, fill_value=self.fill_value)
        self.imp.fit(X)
        self.statistics_ = pd.Series(self.imp.statistics_, index=X.columns)
        return self

    def transform(self, X):
        # assumes X is a DataFrame
        Ximp = self.imp.transform(X)
        Xfilled = pd.DataFrame(Ximp, index=X.index, columns=X.columns)
        return Xfilled


class DFStandardScaler(BaseEstimator, TransformerMixin):
    # StandardScaler but for pandas DataFrames

    def __init__(self):
        self.ss = None
        self.mean_ = None
        self.scale_ = None

    def fit(self, X, y=None):
        self.ss = StandardScaler()
        self.ss.fit(X)
        self.mean_ = pd.Series(self.ss.mean_, index=X.columns)
        self.scale_ = pd.Series(self.ss.scale_, index=X.columns)
        return self

    def transform(self, X):
        # assumes X is a DataFrame
        Xss = self.ss.transform(X)
        Xscaled = pd.DataFrame(Xss, index=X.index, columns=X.columns)
        return Xscaled


class ColumnExtractor(BaseEstimator, TransformerMixin):

    def __init__(self, cols):
        self.cols = cols

    def fit(self, X, y=None):
        # stateless transformer
        return self

    def transform(self, X):
        # assumes X is a DataFrame
        return X[self.cols]


class PCATransformer(BaseEstimator, TransformerMixin):

    def __init__(self, n_components=10):
        self.transformer = None
        self.n_components = n_components

    def fit(self, X, y=None):
        self.transformer = PCA(n_components=self.n_components)
        Xt = X.fillna(-555555)
        self.transformer.fit(Xt)
        return self

    def transform(self, X):
        Xt = X.fillna(-555555)
        Xt = self.transformer.transform(Xt)
        Xt = pd.DataFrame(Xt, index=X.index, columns=[f'PCA_{x}' for x in range(self.n_components)])
        return Xt


class MulticollinearityThreshold(BaseEstimator, TransformerMixin):
    threshold = 0.9
    correlated_features = set()

    def __init__(self, threshold=0.9):
        self.threshold = threshold

    def fit(self, X, y=None):
        self.correlated_features = set()
        numerical_features = list(X.select_dtypes(exclude=[np.object]).columns)
        Xt = X[numerical_features].fillna(-555555)
        correlation_matrix = pd.DataFrame(data=Xt[numerical_features]).corr()
        for i in range(len(correlation_matrix.columns)):
            for j in range(i):
                if abs(correlation_matrix.iloc[i, j]) > self.threshold:
                    self.correlated_features.add(numerical_features[i])
        return self

    def transform(self, X):
        use_columns = set(X.columns) - self.correlated_features
        return X[list(use_columns)]


class GeneratorNumFeatures(BaseEstimator, TransformerMixin):
    features = None

    def __init__(self, features):
        self.features = features

    def fit(self, X, y=None):

        return self

    def transform(self, X):
        operations = [np.sqrt, np.log2, np.square]
        for column in self.features:
            for operation in operations:
                X[f"{column}_{str(operation)}"] = X[column].apply(operation)
        return X
