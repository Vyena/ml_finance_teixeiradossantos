import json
import pickle
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier


SEED = 42

MODELOS_CONFIG = {
    "Random Forest": (
        RandomForestClassifier(random_state=SEED),
        {"n_estimators": [100, 200], "max_depth": [4, 6, None]},
    ),
    "XGBoost": (
        XGBClassifier(random_state=SEED, eval_metric="logloss", verbosity=0),
        {"n_estimators": [100, 200], "max_depth": [3, 5], "learning_rate": [0.05, 0.1]},
    ),
    "Logistic Regression": (
        LogisticRegression(random_state=SEED, max_iter=1000),
        {"C": [0.01, 0.1, 1, 10]},
    ),
}


def train_all_models(
    X_train: np.ndarray,
    y_train: np.ndarray
) -> dict:
    """Entrena todos los modelos de MODELOS_CONFIG con GridSearchCV y retorna los mejores estimadores."""

    best_models = {}

    for name, (model, params) in MODELOS_CONFIG.items():
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=params,
            cv=5,
            scoring="roc_auc",
            n_jobs=-1
        )

        grid_search.fit(X_train, y_train)

        best_models[name] = grid_search.best_estimator_

    return best_models


def evaluate_models(
    models: dict,
    X_test: np.ndarray,
    y_test: np.ndarray
) -> pd.DataFrame:
    """Retorna DataFrame con AUC-ROC de cada modelo ordenado de mayor a menor."""

    results = []

    for name, model in models.items():
        y_proba = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_proba)

        results.append({
            "Modelo": name,
            "AUC": auc
        })

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(by="AUC", ascending=False)

    return results_df


def save_model(
    model,
    path: str,
    metadata: dict
) -> None:
    """Serializa el modelo en .pkl y guarda metadata.json en el mismo directorio."""

    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    model_path = path / "model.pkl"
    metadata_path = path / "metadata.json"

    with open(model_path, "wb") as file:
        pickle.dump(model, file)

    metadata = metadata.copy()
    metadata["saved_at"] = date.today().isoformat()

    with open(metadata_path, "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)