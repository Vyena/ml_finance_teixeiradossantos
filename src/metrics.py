import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def auc_roc(model, X: np.ndarray, y: np.ndarray) -> float:
    """Retorna el AUC-ROC del modelo sobre (X, y)."""

    y_prob = model.predict_proba(X)[:, 1]

    return roc_auc_score(y, y_prob)


def costo_total(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    umbral: float,
    c_fn: float = 500,
    c_fp: float = 100
) -> float:
    """
    Calcula el costo operacional total dado un umbral de clasificación.

    c_fn : costo de Falso Negativo
    c_fp : costo de Falso Positivo
    """

    y_pred = (y_prob >= umbral).astype(int)

    fn = np.sum((y_true == 1) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))

    return fn * c_fn + fp * c_fp


def build_scorecard(
    model_lr,
    woe_tables: dict,
    base_score: int = 300,
    pdo: int = 50
) -> pd.DataFrame:
    """
    Convierte los coeficientes de regresión logística en puntos de scorecard.
    """

    factor = pdo / np.log(2)

    coeficientes = model_lr.coef_[0]
    intercepto = model_lr.intercept_[0]

    n_vars = len(coeficientes)

    scorecard_rows = []

    for i, (feature, table) in enumerate(woe_tables.items()):

        coef = coeficientes[i]

        for _, row in table.iterrows():

            puntos = (
                -(
                    coef * row["woe"] +
                    intercepto / n_vars
                )
            ) * factor + base_score

            scorecard_rows.append({
                "feature": feature,
                "bin": str(row["bin"]),
                "woe": row["woe"],
                "puntos": round(puntos, 2)
            })

    return pd.DataFrame(scorecard_rows)