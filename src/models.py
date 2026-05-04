from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Bidirectional, Dropout, Input
from tensorflow.keras.optimizers import Adam
from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA
from logger_config import setup_logger

logger = setup_logger(__name__)

WINDOW = 20

# Valid params per model — used to filter out irrelevant keys
_RF_PARAMS     = {"n_estimators", "max_depth", "min_samples_split", "max_features",
                  "min_samples_leaf", "random_state"}
_XGB_PARAMS    = {"n_estimators", "max_depth", "learning_rate", "subsample",
                  "colsample_bytree", "reg_alpha", "reg_lambda", "random_state"}
_LINEAR_PARAMS = {"fit_intercept"}
_LSTM_PARAMS   = {"units", "units2", "dropout", "recurrent_dropout", "learning_rate"}
_LGBM_PARAMS   = {"n_estimators", "max_depth", "learning_rate", "num_leaves",
                  "subsample", "colsample_bytree", "reg_alpha", "reg_lambda", "random_state"}
_CATBOOST_PARAMS = {"iterations", "depth", "learning_rate", "l2_leaf_reg",
                    "random_state", "verbose"}
_SVR_PARAMS    = {"kernel", "C", "epsilon", "gamma"}
_KNN_PARAMS    = {"n_neighbors", "weights", "algorithm", "p"}
_PROPHET_PARAMS = {"growth", "changepoint_prior_scale", "seasonality_prior_scale",
                   "yearly_seasonality", "weekly_seasonality", "daily_seasonality"}
_ARIMA_PARAMS  = {"p", "d", "q"}


def get_model(name, **params):
    if name == "linear":
        filtered = {k: v for k, v in params.items() if k in _LINEAR_PARAMS}
        logger.debug(f"Creating LinearRegression | params={filtered}")
        return LinearRegression(**filtered)

    elif name == "rf":
        filtered = {k: v for k, v in params.items() if k in _RF_PARAMS}
        logger.debug(f"Creating RandomForestRegressor | params={filtered}")
        return RandomForestRegressor(**filtered)

    elif name == "xgb":
        filtered = {k: v for k, v in params.items() if k in _XGB_PARAMS}
        logger.debug(f"Creating XGBRegressor | params={filtered}")
        return XGBRegressor(**filtered)

    elif name == "lgbm":
        filtered = {k: v for k, v in params.items() if k in _LGBM_PARAMS}
        filtered.pop("verbose", None)  # force silent — never let param_grid override
        logger.debug(f"Creating LGBMRegressor | params={filtered}")
        return LGBMRegressor(**filtered, verbose=-1)

    elif name == "catboost":
        filtered = {k: v for k, v in params.items() if k in _CATBOOST_PARAMS}
        filtered.pop("verbose", None)  # force silent — never let param_grid override
        logger.debug(f"Creating CatBoostRegressor | params={filtered}")
        return CatBoostRegressor(**filtered, verbose=0)

    elif name == "svr":
        filtered = {k: v for k, v in params.items() if k in _SVR_PARAMS}
        logger.debug(f"Creating SVR | params={filtered}")
        return SVR(**filtered)

    elif name == "knn":
        filtered = {k: v for k, v in params.items() if k in _KNN_PARAMS}
        logger.debug(f"Creating KNeighborsRegressor | params={filtered}")
        return KNeighborsRegressor(**filtered)

    elif name == "lstm":
        filtered = {k: v for k, v in params.items() if k in _LSTM_PARAMS}
        logger.debug(f"Creating Bidirectional LSTM | params={filtered}")
        model = Sequential([
            Input(shape=(WINDOW, 1)),
            Bidirectional(LSTM(
                filtered.get("units", 64),
                return_sequences=True,
                recurrent_dropout=filtered.get("recurrent_dropout", 0.0),
            )),
            Dropout(filtered.get("dropout", 0.0)),
            LSTM(
                filtered.get("units2", 32),
                recurrent_dropout=filtered.get("recurrent_dropout", 0.0),
            ),
            Dropout(filtered.get("dropout", 0.0)),
            Dense(1)
        ])
        model.compile(
            optimizer=Adam(learning_rate=filtered.get("learning_rate", 0.001)),
            loss="mse"
        )
        return model

    elif name == "prophet":
        filtered = {k: v for k, v in params.items() if k in _PROPHET_PARAMS}
        logger.debug(f"Creating Prophet | params={filtered}")
        return Prophet(**filtered)

    elif name == "arima":
        filtered = {k: v for k, v in params.items() if k in _ARIMA_PARAMS}
        logger.debug(f"Creating ARIMA | params={filtered}")
        return ARIMA(**filtered)

    else:
        logger.error(f"Unknown model name: '{name}'")
        raise ValueError(f"Unknown model: '{name}'")


def get_lgbm_model(**params):
    """LightGBM model - faster alternative to XGBoost."""
    filtered = {k: v for k, v in params.items() if k in _LGBM_PARAMS}
    logger.debug(f"Creating LGBMRegressor | params={filtered}")
    return LGBMRegressor(**filtered, verbose=-1)


def get_catboost_model(**params):
    """CatBoost model - excellent gradient boosting with categorical support."""
    filtered = {k: v for k, v in params.items() if k in _CATBOOST_PARAMS}
    logger.debug(f"Creating CatBoostRegressor | params={filtered}")
    return CatBoostRegressor(**filtered)


def get_svr_model(**params):
    """Support Vector Regression - different approach using kernel methods."""
    filtered = {k: v for k, v in params.items() if k in _SVR_PARAMS}
    logger.debug(f"Creating SVR | params={filtered}")
    return SVR(**filtered)


def get_knn_model(**params):
    """K-Nearest Neighbors - simple but effective for local patterns."""
    filtered = {k: v for k, v in params.items() if k in _KNN_PARAMS}
    logger.debug(f"Creating KNeighborsRegressor | params={filtered}")
    return KNeighborsRegressor(**filtered)


def get_prophet_model(**params):
    """Prophet model - Facebook's forecasting library for business time series."""
    filtered = {k: v for k, v in params.items() if k in _PROPHET_PARAMS}
    logger.debug(f"Creating Prophet | params={filtered}")
    return Prophet(**filtered)


def get_arima_model(**params):
    """ARIMA model - classic statistical time series forecasting."""
    filtered = {k: v for k, v in params.items() if k in _ARIMA_PARAMS}
    logger.debug(f"Creating ARIMA | params={filtered}")
    return ARIMA(**filtered)
