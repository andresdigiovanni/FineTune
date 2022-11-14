import lightgbm as lgb
import numpy as np
import sklearn.datasets
import sklearn.metrics
from sklearn.model_selection import train_test_split

from finetune import Finetune


def lgbm_objective(params, dtrain, valid_x, valid_y):
    gbm = lgb.train(params, dtrain)
    preds = gbm.predict(valid_x)
    pred_labels = np.rint(preds)

    return sklearn.metrics.accuracy_score(valid_y, pred_labels)


if __name__ == "__main__":
    param_grid = {
        "objective": "binary",
        "metric": "binary_logloss",
        "verbosity": -1,
        "boosting_type": "gbdt",
        "feature_pre_filter": False,
        "lambda_l1": ("float_log", 1e-8, 10.0),
        "lambda_l2": ("float_log", 1e-8, 10.0),
        "num_leaves": ("int", 2, 256),
        "feature_fraction": ("float", 0.4, 1.0),
        "bagging_fraction": ("float", 0.4, 1.0),
        "bagging_freq": ("int", 1, 7),
        "min_child_samples": ("int", 5, 100),
        "max_depth": ("int", 3, 50),
    }

    data, target = sklearn.datasets.load_breast_cancer(return_X_y=True)
    train_x, valid_x, train_y, valid_y = train_test_split(data, target, test_size=0.25)
    dtrain = lgb.Dataset(train_x, label=train_y)

    finetune = Finetune(param_grid, objective="maximize", study_name="Breast cancer")
    finetune.optimize(
        lambda trial: lgbm_objective(trial, dtrain, valid_x, valid_y),
        n_trials=50,
    )

    score = finetune.get_best_score()
    print(f"\nScore: {score}")

    finetune.plot_optimization_history()
    finetune.plot_parallel_coordinate()
    finetune.plot_parameter_history("lambda_l1")
