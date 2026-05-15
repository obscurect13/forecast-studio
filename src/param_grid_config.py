param_grid = {
    "rf": [
        {"n_estimators": 100, "max_depth": None, "min_samples_split": 2},
        {"n_estimators": 200, "max_depth": 10, "min_samples_split": 5},
        {"n_estimators": 300, "max_depth": 20, "min_samples_split": 10, "max_features": "sqrt"},
    ],
    "xgb": [
        {"n_estimators": 100, "max_depth": 3, "learning_rate": 0.1, "subsample": 0.8},
        {"n_estimators": 200, "max_depth": 5, "learning_rate": 0.05, "subsample": 0.7, "colsample_bytree": 0.8},
        {"n_estimators": 300, "max_depth": 6, "learning_rate": 0.01, "subsample": 0.6, "reg_alpha": 0.1, "reg_lambda": 1.0},
    ],
    "lgbm": [
        {"n_estimators": 100, "max_depth": 3, "learning_rate": 0.1, "num_leaves": 31},
        {"n_estimators": 200, "max_depth": 5, "learning_rate": 0.05, "num_leaves": 63, "subsample": 0.8},
        {"n_estimators": 300, "max_depth": 7, "learning_rate": 0.01, "num_leaves": 127, "reg_alpha": 0.1},
    ],
    "catboost": [
        {"iterations": 100, "depth": 6, "learning_rate": 0.1},
        {"iterations": 200, "depth": 8, "learning_rate": 0.05, "l2_leaf_reg": 3},
        {"iterations": 300, "depth": 10, "learning_rate": 0.01, "l2_leaf_reg": 5},
    ],
    "svr": [
        {"kernel": "rbf", "C": 1.0, "epsilon": 0.1},
        {"kernel": "rbf", "C": 10.0, "epsilon": 0.01},
        {"kernel": "linear", "C": 1.0, "epsilon": 0.1},
        {"kernel": "poly", "C": 1.0, "epsilon": 0.1, "degree": 2},
    ],
    "knn": [
        {"n_neighbors": 3, "weights": "uniform"},
        {"n_neighbors": 5, "weights": "distance"},
        {"n_neighbors": 7, "weights": "uniform", "algorithm": "auto"},
    ],
    "linear": [
        {},
        {"fit_intercept": False},
    ],
    "lstm": [
        {"units": 32, "units2": 16, "dropout": 0.1, "learning_rate": 0.001},
        {"units": 64, "units2": 32, "dropout": 0.2, "learning_rate": 0.001},
        {"units": 128, "units2": 64, "dropout": 0.3, "learning_rate": 0.0005, "recurrent_dropout": 0.1},
    ],
    # Prophet disabled in Docker/Linux deployment due to Stan backend
    # incompatibility with Python 3.12. Works locally via conda.
    # "prophet": [
    #     {"growth": "linear", "changepoint_prior_scale": 0.05, "seasonality_prior_scale": 10},
    #     {"growth": "linear", "changepoint_prior_scale": 0.1, "seasonality_prior_scale": 5, "yearly_seasonality": True},
    #     {"growth": "logistic", "changepoint_prior_scale": 0.05, "seasonality_prior_scale": 10},
    # ],
    "arima": [
        {"p": 1, "d": 1, "q": 1},
        {"p": 2, "d": 1, "q": 2},
        {"p": 3, "d": 1, "q": 1},
    ],
}