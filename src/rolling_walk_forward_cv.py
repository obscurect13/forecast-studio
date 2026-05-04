from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from logger_config import setup_logger
import numpy as np

logger = setup_logger(__name__)


def rolling_cv(X, y, model_fn, params, n_splits=5, lstm=False):
    fold_size = len(X) // (n_splits + 1)
    mse_scores, mae_scores, r2_scores = [], [], []

    logger.debug(f"Rolling CV | n_splits={n_splits} | fold_size={fold_size} | lstm={lstm}")

    for i in range(1, n_splits + 1):
        split = i * fold_size

        X_train = X[:split]
        y_train = y[:split]
        X_test  = X[split:split + fold_size]
        y_test  = y[split:split + fold_size]

        model = model_fn(**params)

        if lstm:
            model.fit(X_train, y_train, epochs=5, verbose=0)
            preds = model.predict(X_test, verbose=0).flatten()
        else:
            model.fit(X_train.reshape(X_train.shape[0], -1), y_train)
            preds = model.predict(X_test.reshape(X_test.shape[0], -1))

        mse = mean_squared_error(y_test, preds)
        mae = mean_absolute_error(y_test, preds)
        r2  = r2_score(y_test, preds)

        mse_scores.append(mse)
        mae_scores.append(mae)
        r2_scores.append(r2)

        logger.debug(f"  Fold {i}/{n_splits} → MSE={mse:.6f} | MAE={mae:.6f} | R²={r2:.4f}")

    return {
        "mse": float(np.mean(mse_scores)),
        "mae": float(np.mean(mae_scores)),
        "r2":  float(np.mean(r2_scores)),
    }
