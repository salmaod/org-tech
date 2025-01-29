import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def train_and_evaluate(model, X_train, X_test, y_train, y_test):
    """
    Entraîner un modèle, effectuer des prédictions et calculer les métriques de performance.
    """
    model.fit(X_train, y_train)
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    metrics = {
        "MAE (Train)": mean_absolute_error(y_train, y_pred_train),
        "MAE (Test)": mean_absolute_error(y_test, y_pred_test),
        "RMSE (Train)": np.sqrt(mean_squared_error(y_train, y_pred_train)),
        "RMSE (Test)": np.sqrt(mean_squared_error(y_test, y_pred_test)),
        "R² (Train)": r2_score(y_train, y_pred_train),
        "R² (Test)": r2_score(y_test, y_pred_test),
    }

    return model, metrics

def train_decision_tree(X_train, X_test, y_train, y_test, max_depth=None):
    """
    Entraîner un arbre de décision (avec ou sans limitation de profondeur) et calculer les métriques.
    """
    from sklearn.tree import DecisionTreeRegressor
    dt_model = DecisionTreeRegressor(max_depth=max_depth, random_state=42)
    return train_and_evaluate(dt_model, X_train, X_test, y_train, y_test)

def ensemble_learning(rf_model, xgb_model, lgbm_model, X_test, y_test, weights):
    rf_predictions = rf_model.predict(X_test)
    xgb_predictions = xgb_model.predict(X_test)
    lgbm_predictions = lgbm_model.predict(X_test)

    ensemble_predictions = (
        weights["rf"] * rf_predictions +
        weights["xgb"] * xgb_predictions +
        weights["lgbm"] * lgbm_predictions
    )

    metrics = {
        "MAE": mean_absolute_error(y_test, ensemble_predictions),
        "RMSE": np.sqrt(mean_squared_error(y_test, ensemble_predictions)),
        "R2": r2_score(y_test, ensemble_predictions),
    }

    return metrics, ensemble_predictions
