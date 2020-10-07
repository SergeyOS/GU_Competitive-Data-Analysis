import numpy as np
import pandas as pd
from sacred import Experiment
from sacred.observers import FileStorageObserver
from utility import dataset_function as reader
from sklearn.pipeline import Pipeline
from FeatureEngineering import client_profile
from FeatureEngineering import payment_stats
from FeatureEngineering import app_history
from FeatureEngineering import warp_catboost
from FeatureEngineering import features_transform
from FeatureEngineering import features_selection
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

ex = Experiment("GeekBrains Competitive Data Analysis")

ex.observers.append(FileStorageObserver(basedir="Experiments/runs", source_dir='Experiments/source',
                                        resource_dir='Experiments/resource', template='custom/template.txt'))


@ex.config
def cfg():
    seed = 44452
    params = {
        "n_estimators": 2000,
        "learning_rate": 0.01, #0.01
        "loss_function": "Logloss",
        "eval_metric": "AUC",
        "task_type": "CPU",
        "max_bin": 20,
        "verbose": 10,
        "max_depth": 6,
        "l2_leaf_reg": 20,
        "early_stopping_rounds": 50, #50
        "random_seed": seed
    }

    KFold_split = 5
    percent_column = None
    pca_component = 15

@ex.capture
def random_seed(seed):
    np.random.seed(seed)
    return seed


@ex.capture
def get_pipeline(params, KFold_split, seed, percent_column,  pca_component):
    print(f'Параметры модели {params}')
    base_model = warp_catboost.WarpCatboost(params=params, ex=ex, KFold_split=KFold_split)
    client_info = client_profile.ClientProfile('client_profile.csv', ex=ex)
    payment_info = payment_stats.PaymentsStat('payments.csv', ex=ex)
    selector = features_selection.FeaturesSelector(percent_column=percent_column, seed=seed)
    transform_features = features_transform.FeaturesTransform(pca_component)
    app_history_stats = app_history.AppHistory('applications_history.csv', ex)
    return Pipeline(steps=[
        ('client_info', client_info),
        ('payment_info', payment_info),
        ('app_history_stats', app_history_stats),
        ('transform_features', transform_features),
        ('selector', selector),
        ('base_model', base_model)], verbose=1)


@ex.capture
def save_prediction(estimator, test, test_id):
    y_pred = estimator.predict_proba(test)

    y_pred = pd.DataFrame({
        "APPLICATION_NUMBER": test_id,
        "TARGET": y_pred})
    y_pred.to_csv(ex.open_resource("result/baseline_submit.csv", mode='w'), index=False)


@ex.main
def run():
    seed = random_seed()
    train = reader.get_input("train.csv", ex)
    test = reader.get_input("test.csv", ex)
    target, test_id = train["target"], test["application_number"]
    train = train.drop(["target"], axis=1)

    x_train, x_valid, y_train, y_valid = train_test_split(train, target, train_size=0.8, random_state=seed,
                                                          shuffle=True, stratify=target)

    model = get_pipeline()
    print(model)
    model.fit(x_train, y_train)
    valid_score = roc_auc_score(y_valid, model.predict_proba(x_valid))
    print(f"Score на отложенной выборке = {round(valid_score, 4)}")
    ex.log_scalar('Score last partion', valid_score, 'last')
    save_prediction(model, test, test_id)
    del model


if __name__ == "__main__":
    ex.run()
