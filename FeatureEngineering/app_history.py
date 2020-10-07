from utility import dataset_function as reader
import pandas as pd
import numpy as np
from sklearn.base import TransformerMixin
from sacred import Experiment


def fill_categorical_features(data):
    for column in data.columns:
        nunique_value = data[column].nunique()
        if nunique_value < 10:
            data[column] = data[column].astype(object)
    for column in data.select_dtypes(include=[np.object]):
        data[column] = data[column].astype('category')
        if data[column].isnull().any():
            data[column] = data[column].cat.add_categories(['MISSING'])
            data[column] = data[column].fillna('MISSING')
    return data


def get_categorical_stat(data):
    columns = ['name_contract_type',
               'name_contract_status', 'name_payment_type', 'code_reject_reason',
               'name_type_suite', 'name_client_type', 'name_goods_category',
               'name_portfolio', 'name_product_type', 'name_yield_group',
               'nflag_insured_on_approval']
    stat_categorical = pd.DataFrame(data=data['application_number'], columns=['application_number'])
    for column in columns:
        stat_categorical = pd.concat([stat_categorical, pd.get_dummies(data[column], prefix=str(column))], axis=1);
    stat_categorical['prev_application_number'] = 1
    stat_categorical = stat_categorical.groupby(by=['application_number']).sum()
    stat_categorical.columns = [f"{x}_count" for x in stat_categorical.columns]
    return stat_categorical


def get_stat_numbers_by_features(data, feature=None, value_feature=None):
    columns = ['amount_annuity', 'amount_credit', 'amount_goods_payment',
               'amount_payment', 'amt_application', 'application_number',
               'cnt_payment', 'days_decision', 'days_first_drawing',
               'days_first_due', 'days_last_due', 'days_last_due_1st_version',
               'days_termination']
    if value_feature is None:
        stat_by_feature = data.loc[:, columns].copy()
        value_feature = 'All'
    else:
        stat_by_feature = data.loc[data[feature] == value_feature, columns].copy()
    stat_by_feature['diff_amount_goods_amount_credit'] = stat_by_feature['amount_goods_payment'] - stat_by_feature[
        'amount_credit']
    stat_by_feature = stat_by_feature.groupby(by="application_number").agg(
        ["min", "max", "mean", np.nanmedian, "var"]).reset_index()
    stat_by_feature.columns = ["_".join(x) for x in stat_by_feature.columns.ravel()]
    stat_by_feature.columns = [f"{value_feature}_{x}" for x in stat_by_feature.columns]
    stat_by_feature = stat_by_feature.rename(columns={f'{value_feature}_application_number_': 'application_number'})
    return stat_by_feature

def add_preffix(data, prefix):
    data.columns = [f"{prefix}_{x}" for x in data.columns]
    data = data.rename(columns={f'{prefix}_application_number': 'application_number'})
    return data

class AppHistory(TransformerMixin):
    stat_categorical: pd.DataFrame
    stat_by_Canceled: pd.DataFrame
    stat_by_Approved: pd.DataFrame
    stat_by_all: pd.DataFrame

    def __init__(self, filename: str, ex: Experiment):
        data = reader.get_input(filename, ex)
        data = fill_categorical_features(data)
        self.stat_categorical = add_preffix(get_categorical_stat(data), 'ah')

        self.stat_by_Canceled = add_preffix(get_stat_numbers_by_features(data, 'name_contract_status', 'Canceled'), 'ah')
        self.stat_by_Approved = add_preffix(get_stat_numbers_by_features(data, 'name_contract_status', 'Approved'), 'ah')
        self.stat_by_all = add_preffix(get_stat_numbers_by_features(data, 'All', None), 'ah')

        del data

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        assert isinstance(X, pd.DataFrame)
        Xt = X.merge(self.stat_categorical, how="left", on="application_number")
        Xt = Xt.merge(self.stat_by_Canceled, how="left", on="application_number")
        Xt = Xt.merge(self.stat_by_Approved, how="left", on="application_number")
        Xt = Xt.merge(self.stat_by_all, how="left", on="application_number")
        return Xt
