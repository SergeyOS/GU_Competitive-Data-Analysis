import pandas as pd
from sacred import Experiment


def get_input(data_path: str, ex: Experiment = None) -> pd.DataFrame:
    """
    Считывание данных и вывод основной информации о наборе данных.

    Parameters
    ----------
    data_path: str
        Название файла.
    ex: Experiment=None
        Объект эксперимента

    Returns
    -------
    data: pandas.core.frame.DataFrame
        Загруженный набор данных в pandas.DataFrame

    """
    base_path = "data"
    path_file = f"{base_path}/{data_path}"
    if ex is None:
        data = pd.read_csv(path_file)
    else:
        data = pd.read_csv(ex.open_resource(path_file))
    data.columns = [col.lower() for col in data.columns]
    print(f"{data_path}: shape = {data.shape[0]} rows, {data.shape[1]} cols")

    return data
