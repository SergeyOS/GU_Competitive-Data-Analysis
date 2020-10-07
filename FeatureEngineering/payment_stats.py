from utility import dataset_function as reader
import pandas as pd
import numpy as np
from sklearn.base import TransformerMixin
from sacred import Experiment


def create_payments_features(X: pd.DataFrame, copy: bool = True) -> pd.DataFrame:
    """
    Создание признаков на основе истории платежей/неплатежей

    Parameters
    ----------
    X: pandas.core.frame.DataFrame
        Матрица признаков с исходной историей платежей клиентов.

    copy: bool, optional, default = True
        Флаг использования копии датафрейма X.
        Опциональный параметр, по умолчанию, равен True.

    Returns
    -------
    X_transformed: pandas.core.frame.DataFrame
        Статистика платежей.

    """
    if copy:
        X = X.copy()

    # различие между датами платежа и сроком платежа
    X['diff_days'] = X['days_entry_payment'] - X['days_instalment']
    # различие между платежом и обязательством
    X['diff_amt'] = X['amt_instalment'] - X['amt_payment']
    # сумма просрочки долга
    X['overdue_debt'] = X['diff_amt']
    X.loc[X['overdue_debt'] < 0, 'overdue_debt'] = 0
    # факт просрочки более N дней
    X['flag_overdue_debt'] = (np.logical_or(X['diff_days'] > 0, X['diff_amt'] > 0)).astype(int)
    X['flag_overdue_debt_30'] = (X['diff_days'] > 30).astype(int)
    X['flag_overdue_debt_90'] = (X['diff_days'] > 90).astype(int)
    X['flag_overdue_debt_181'] = (X['diff_days'] > 181).astype(int)
    X['flag_overdue_debt_365'] = (X['diff_days'] > 365).astype(int)
    # Отношение непогашенного долга к обязательству
    X['ratio_overdue_debt'] = X['diff_amt'] / X['amt_instalment']
    X.loc[X['ratio_overdue_debt'] < 0, 'ratio_overdue_debt'] = -1  # переплаты более обязательства приводим к -1
    # Статистики
    payments_app_info = X.groupby(by="application_number").agg(
        amt_payment_min=pd.NamedAgg(column='amt_payment', aggfunc='min'),
        amt_payment_max=pd.NamedAgg(column='amt_payment', aggfunc='max'),
        amt_payment_mean=pd.NamedAgg(column='amt_payment', aggfunc='mean'),
        amt_payment_median=pd.NamedAgg(column='amt_payment', aggfunc='median'),
        diff_days_min=pd.NamedAgg(column='diff_days', aggfunc='min'),
        diff_days_max=pd.NamedAgg(column='diff_days', aggfunc='max'),
        diff_days_count=pd.NamedAgg(column='diff_days', aggfunc='count'),
        flag_overdue_debt=pd.NamedAgg(column='flag_overdue_debt', aggfunc='sum'),
        flag_overdue_debt_30=pd.NamedAgg(column='flag_overdue_debt_30', aggfunc='sum'),
        flag_overdue_debt_90=pd.NamedAgg(column='flag_overdue_debt_90', aggfunc='sum'),
        flag_overdue_debt_181=pd.NamedAgg(column='flag_overdue_debt_181', aggfunc='sum'),
        flag_overdue_debt_365=pd.NamedAgg(column='flag_overdue_debt_365', aggfunc='sum'),
        ratio_overdue_debt_mean=pd.NamedAgg(column='ratio_overdue_debt', aggfunc='mean'),
        ratio_overdue_debt_min=pd.NamedAgg(column='ratio_overdue_debt', aggfunc='min'),
        ratio_overdue_debt_max=pd.NamedAgg(column='ratio_overdue_debt', aggfunc='max'),
        overdue_debt_mean=pd.NamedAgg(column='overdue_debt', aggfunc='mean')
    ).reset_index()

    return payments_app_info


def add_preffix(data, prefix):
    data.columns = [f"{prefix}_{x}" for x in data.columns]
    data = data.rename(columns={f'{prefix}_application_number': 'application_number'})
    return data


class PaymentsStat(TransformerMixin):
    dataset: pd.DataFrame

    def __init__(self, filename: str, ex: Experiment):
        self.dataset = reader.get_input(filename, ex)
        self.dataset = add_preffix(create_payments_features(self.dataset, False), 'p')

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        assert isinstance(X, pd.DataFrame)
        Xt = X.merge(self.dataset, how="left", on="application_number")
        return Xt
