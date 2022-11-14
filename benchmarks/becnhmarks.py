import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
import sklearn.datasets
import sklearn.metrics
from optimization_test_functions import (
    Abs,
    Ackley,
    AckleyTest,
    Eggholder,
    Fletcher,
    Griewank,
    Michalewicz,
    Noises,
    Penalty2,
    Quartic,
    Rastrigin,
    Rosenbrock,
    Scheffer,
    SchwefelAbs,
    SchwefelDouble,
    SchwefelMax,
    SchwefelSin,
    Sphere,
    Stairs,
    Transformation,
    Weierstrass,
)
from sklearn.model_selection import train_test_split

from finetune import Finetune


def finetune_func_objective(params, func):
    return func(np.array([x for x in params.values()]))


def optuna_func_objective(trial, func, param_grid):
    params = {}

    for k, v in param_grid.items():
        params[k] = trial.suggest_float(k, v[0], v[1])

    return func(np.array([x for x in params.values()]))


def test_func(func, param_grid, n_trials, cv):
    func = Transformation(func, noise_generator=Noises.uniform(-0.1, 0.5))

    objective = "minimize"
    finetune_param_grid = {}

    for k, v in param_grid.items():
        finetune_param_grid[k] = ("float", v[0], v[1])

    scores_finetune = []
    best_trial_number_finetune = []
    scores_optuna = []
    best_trial_number_optuna = []

    for i in range(cv):
        # Finetune
        finetune = Finetune(finetune_param_grid, objective=objective)
        finetune.optimize(
            lambda trial: finetune_func_objective(trial, func), n_trials=n_trials
        )
        scores_finetune.append(finetune.get_best_score())
        best_trial_number_finetune.append(finetune.get_best_trial_number())

        # Optuna
        optuna_study = optuna.create_study(direction=objective)
        optuna_study.optimize(
            lambda trial: optuna_func_objective(trial, func, param_grid),
            n_trials=n_trials,
        )
        scores_optuna.append(optuna_study.best_value)
        best_trial_number_optuna.append(optuna_study.best_trial.number)

    mean_finetune = round(np.mean(scores_finetune), 3)
    std_finetune = round(np.std(scores_finetune), 3)
    best_trial_number_finetune = round(np.mean(best_trial_number_finetune))

    mean_optuna = round(np.mean(scores_optuna), 3)
    std_optuna = round(np.std(scores_optuna), 3)
    best_trial_number_optuna = round(np.mean(best_trial_number_optuna))

    return (
        mean_finetune,
        std_finetune,
        best_trial_number_finetune,
        mean_optuna,
        std_optuna,
        best_trial_number_optuna,
        objective,
    )


def tests_optimizations_funcs(df_results, dim, n_trials, cv):

    # best solution: f[0,0] = 0.0
    df_results.loc["Abs"] = test_func(
        Abs(dim=dim),
        {f"x{i}": (-10, 10) for i in range(dim)},
        n_trials=n_trials,
        cv=cv,
    )
    # best solution: f[0,0] = 0.0
    df_results.loc["Ackley"] = test_func(
        Ackley(dim=dim),
        {f"x{i}": (-3, 3) for i in range(dim)},
        n_trials=n_trials,
        cv=cv,
    )
    df_results.loc["AckleyTest"] = test_func(
        AckleyTest(dim=dim),
        {f"x{i}": (-30, 30) for i in range(dim)},
        n_trials=n_trials,
        cv=cv,
    )
    df_results.loc["Eggholder"] = test_func(
        Eggholder(dim=dim),
        {f"x{i}": (-500, 500) for i in range(dim)},
        n_trials=n_trials,
        cv=cv,
    )
    # best solution: f[1.5386377,-1.87784459] = 0.0
    df_results.loc["Fletcher"] = test_func(
        Fletcher(dim=dim, seed=1),
        {f"x{i}": (-3, 3) for i in range(dim)},
        n_trials=n_trials,
        cv=cv,
    )
    # best solution: f[0,0] = 0.0
    df_results.loc["Griewank"] = test_func(
        Griewank(dim=dim),
        {f"x{i}": (-600, 600) for i in range(dim)},
        n_trials=n_trials,
        cv=cv,
    )
    df_results.loc["Michalewicz"] = test_func(
        Michalewicz(m=10),
        {f"x{i}": (0, 3) for i in range(dim)},
        n_trials=n_trials,
        cv=cv,
    )
    # best solution: f[1,1] = 0.0
    df_results.loc["Penalty2"] = test_func(
        Penalty2(dim=dim, a=5, k=100, m=4),
        {f"x{i}": (-50, 50) for i in range(dim)},
        n_trials=n_trials,
        cv=cv,
    )
    # best solution: f[0,0] = 0.0
    df_results.loc["Quartic"] = test_func(
        Quartic(dim=dim),
        {f"x{i}": (-1.5, 1.5) for i in range(dim)},
        n_trials=n_trials,
        cv=cv,
    )
    # best solution: f[0,0] = 0.0
    df_results.loc["Rastrigin"] = test_func(
        Rastrigin(dim=dim),
        {f"x{i}": (-5, 5) for i in range(dim)},
        n_trials=n_trials,
        cv=cv,
    )
    df_results.loc["Rosenbrock"] = test_func(
        Rosenbrock(dim=dim),
        {f"x{i}": (-2, 2) for i in range(dim)},
        n_trials=n_trials,
        cv=cv,
    )
    # best solution: f[0,0] = 0.0
    df_results.loc["Scheffer"] = test_func(
        Scheffer(dim=dim),
        {f"x{i}": (-7, 7) for i in range(dim)},
        n_trials=n_trials,
        cv=cv,
    )
    # best solution: f[0,0] = 0.0
    df_results.loc["SchwefelAbs"] = test_func(
        SchwefelAbs(dim=dim),
        {f"x{i}": (-10, 10) for i in range(dim)},
        n_trials=n_trials,
        cv=cv,
    )
    # best solution: f[0,0] = 0.0
    df_results.loc["SchwefelDouble"] = test_func(
        SchwefelDouble(dim=dim),
        {f"x{i}": (-70, 70) for i in range(dim)},
        n_trials=n_trials,
        cv=cv,
    )
    # best solution: f[0,0] = 0.0
    df_results.loc["SchwefelMax"] = test_func(
        SchwefelMax(dim=dim),
        {f"x{i}": (-100, 100) for i in range(dim)},
        n_trials=n_trials,
        cv=cv,
    )
    # best solution: f[420.9687,420.9687] = -838.0
    df_results.loc["SchwefelSin"] = test_func(
        SchwefelSin(dim=dim),
        {f"x{i}": (-500, 500) for i in range(dim)},
        n_trials=n_trials,
        cv=cv,
    )
    df_results.loc["Sphere"] = test_func(
        Sphere(dim=dim, degree=2),
        {f"x{i}": (-5, 5) for i in range(dim)},
        n_trials=n_trials,
        cv=cv,
    )
    # best solution: f[0,0] = 0.0
    df_results.loc["Stairs"] = test_func(
        Stairs(dim=dim),
        {f"x{i}": (-6, 6) for i in range(dim)},
        n_trials=n_trials,
        cv=cv,
    )
    # best solution: f[0,0] = 0.0
    df_results.loc["Weierstrass"] = test_func(
        Weierstrass(dim=dim, a=0.5, b=3, kmax=20),
        {f"x{i}": (-0.5, 0.5) for i in range(dim)},
        n_trials=n_trials,
        cv=cv,
    )


def lgbm_objective(params, dtrain, valid_x, valid_y):
    gbm = lgb.train(params, dtrain)
    preds = gbm.predict(valid_x)
    pred_labels = np.rint(preds)

    if params["objective"] == "binary":
        score = sklearn.metrics.accuracy_score(valid_y, pred_labels)
    else:
        score = sklearn.metrics.mean_squared_error(valid_y, pred_labels)

    return score


def finetune_lgbm_objective(params, dtrain, valid_x, valid_y):
    return lgbm_objective(params, dtrain, valid_x, valid_y)


def optuna_lgbm_objective(trial, dtrain, valid_x, valid_y, param_grid):
    params = {}

    for k, v in param_grid.items():
        if type(v) is tuple:
            log = "log" in v[0]

            if "float" in v[0]:
                params[k] = trial.suggest_float(k, v[1], v[2], log=log)
            else:
                params[k] = trial.suggest_int(k, v[1], v[2], log=log)
        else:
            params[k] = v

    return lgbm_objective(params, dtrain, valid_x, valid_y)


def test_lgbm(data, target, param_grid, n_trials, cv):
    train_x, valid_x, train_y, valid_y = train_test_split(data, target, test_size=0.25)
    dtrain = lgb.Dataset(train_x, label=train_y)

    objective = "maximize" if param_grid["objective"] == "binary" else "minimize"

    finetune_param_grid = {}

    for k, v in param_grid.items():
        if type(v) is tuple:
            type_v, min_v, max_v = v[0], v[1], v[2]
            finetune_param_grid[k] = (type_v, min_v, max_v)
        else:
            finetune_param_grid[k] = v

    scores_finetune = []
    best_trial_number_finetune = []
    scores_optuna = []
    best_trial_number_optuna = []

    for i in range(cv):
        # Finetune
        finetune = Finetune(finetune_param_grid, objective=objective)
        finetune.optimize(
            lambda trial: finetune_lgbm_objective(trial, dtrain, valid_x, valid_y),
            n_trials=n_trials,
        )
        scores_finetune.append(finetune.get_best_score())
        best_trial_number_finetune.append(finetune.get_best_trial_number())

        # Optuna
        optuna_study = optuna.create_study(direction=objective)
        optuna_study.optimize(
            lambda trial: optuna_lgbm_objective(
                trial, dtrain, valid_x, valid_y, param_grid
            ),
            n_trials=n_trials,
        )
        scores_optuna.append(optuna_study.best_value)
        best_trial_number_optuna.append(optuna_study.best_trial.number)

    mean_finetune = round(np.mean(scores_finetune), 3)
    std_finetune = round(np.std(scores_finetune), 3)
    best_trial_number_finetune = round(np.mean(best_trial_number_finetune))

    mean_optuna = round(np.mean(scores_optuna), 3)
    std_optuna = round(np.std(scores_optuna), 3)
    best_trial_number_optuna = round(np.mean(best_trial_number_optuna))

    return (
        mean_finetune,
        std_finetune,
        best_trial_number_finetune,
        mean_optuna,
        std_optuna,
        best_trial_number_optuna,
        objective,
    )


def tests_lgbm_funcs(df_results, n_trials, cv):
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

    data, target = sklearn.datasets.load_iris(return_X_y=True)
    param_grid["objective"] = "binary"
    param_grid["metric"] = "binary_logloss"
    df_results.loc["lgbm - iris"] = test_lgbm(data, target, param_grid, n_trials, cv)
    data, target = sklearn.datasets.load_diabetes(return_X_y=True)
    param_grid["objective"] = "regression"
    param_grid["metric"] = "rmse"
    df_results.loc["lgbm - diabetes"] = test_lgbm(
        data, target, param_grid, n_trials, cv
    )
    data, target = sklearn.datasets.load_digits(return_X_y=True)
    param_grid["objective"] = "binary"
    param_grid["metric"] = "binary_logloss"
    df_results.loc["lgbm - digits"] = test_lgbm(data, target, param_grid, n_trials, cv)
    data, target = sklearn.datasets.load_wine(return_X_y=True)
    param_grid["objective"] = "binary"
    param_grid["metric"] = "binary_logloss"
    df_results.loc["lgbm - wine"] = test_lgbm(data, target, param_grid, n_trials, cv)
    data, target = sklearn.datasets.load_breast_cancer(return_X_y=True)
    param_grid["objective"] = "binary"
    param_grid["metric"] = "binary_logloss"
    df_results.loc["lgbm - breast cancer"] = test_lgbm(
        data, target, param_grid, n_trials, cv
    )


if __name__ == "__main__":
    n_trials = 50
    cv = 3
    dim = 5

    columns = [
        "Finetune mean",
        "Finetune std",
        "Finetune n trial",
        "Optuna mean",
        "Optuna std",
        "Optuna n trial",
        "Direction",
    ]

    df_results = pd.DataFrame(columns=columns)

    tests_optimizations_funcs(df_results, dim, n_trials, cv)
    tests_lgbm_funcs(df_results, n_trials, cv)

    # Compare methods
    comparative = (
        (df_results["Finetune mean"] > df_results["Optuna mean"])
        & (df_results["Direction"] == "maximize")
    ) + (
        (df_results["Finetune mean"] < df_results["Optuna mean"])
        & (df_results["Direction"] == "minimize")
    )

    total = len(comparative)
    gains = sum(comparative)
    print(f"\nFinetune win ratio: {gains/total}")

    # Resume
    df_results["Finetune"] = (
        df_results["Finetune mean"].astype(str)
        + " (+/-"
        + df_results["Finetune std"].astype(str)
        + ")"
    )
    df_results["Optuna"] = (
        df_results["Optuna mean"].astype(str)
        + " (+/-"
        + df_results["Optuna std"].astype(str)
        + ")"
    )

    for x in columns:
        del df_results[x]

    print("\n--------------------")
    print(df_results)
