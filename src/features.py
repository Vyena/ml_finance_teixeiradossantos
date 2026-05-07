
import numpy as np
import pandas as pd


def compute_woe_iv(
    df: pd.DataFrame,
    feature: str,
    target: str,
    bins: int = 10
) -> tuple[pd.DataFrame, float]:
    """
    Calcula WoE e IV para una variable numérica continua.

    Returns
    -------
    woe_table : DataFrame con columnas [bin, n_events, n_non_events, woe, iv_bin]
    iv        : Information Value total de la variable
    """

    # Discretiza la variable en bins
    temp_df = df[[feature, target]].copy()
    temp_df["bin"] = pd.qcut(temp_df[feature], q=bins, duplicates="drop")

    # Agrupa por bins
    grouped = temp_df.groupby("bin")[target].agg(["count", "sum"])

    grouped.columns = ["total", "n_events"]
    grouped["n_non_events"] = grouped["total"] - grouped["n_events"]

    total_events = grouped["n_events"].sum()
    total_non_events = grouped["n_non_events"].sum()

    grouped["dist_events"] = grouped["n_events"] / total_events
    grouped["dist_non_events"] = grouped["n_non_events"] / total_non_events

    # Evita divisão por zero
    grouped["dist_events"] = grouped["dist_events"].replace(0, 0.0001)
    grouped["dist_non_events"] = grouped["dist_non_events"].replace(0, 0.0001)

    # Calcula WoE
    grouped["woe"] = np.log(
        grouped["dist_events"] / grouped["dist_non_events"]
    )

    # Calcula IV por bin
    grouped["iv_bin"] = (
        grouped["dist_events"] - grouped["dist_non_events"]
    ) * grouped["woe"]

    # IV total
    iv = grouped["iv_bin"].sum()

    woe_table = grouped.reset_index()

    return woe_table, iv


def select_features_by_iv(
    df: pd.DataFrame,
    target: str,
    threshold: float = 0.1
) -> list[str]:
    """Retorna la lista de variables con IV >= threshold."""

    selected_features = []

    numeric_columns = df.select_dtypes(include=np.number).columns

    for col in numeric_columns:
        if col != target:

            try:
                _, iv = compute_woe_iv(df, col, target)

                if iv >= threshold:
                    selected_features.append(col)

            except:
                pass

    return selected_features


def build_woe_tables(
    df: pd.DataFrame,
    features: list[str],
    target: str
) -> dict[str, pd.DataFrame]:
    """Genera el diccionario {feature: woe_table} para las variables seleccionadas."""

    woe_tables = {}

    for feature in features:
        woe_table, _ = compute_woe_iv(df, feature, target)
        woe_tables[feature] = woe_table

    return woe_tables


def transform_woe(
    df: pd.DataFrame,
    woe_tables: dict[str, pd.DataFrame]
    
) -> pd.DataFrame:
    """Reemplaza cada feature por su valor WoE según la tabla de entrenamiento."""

    transformed_df = pd.DataFrame(index=df.index)

    for feature, table in woe_tables.items():

        bins = table["bin"]
        woe_values = table["woe"]

        temp_bins = pd.qcut(
            df[feature],
            q=len(bins),
            duplicates="drop"
        )

        mapping = dict(zip(bins.astype(str), woe_values))

        transformed_df[f"{feature}_woe"] = (
            temp_bins.astype(str).map(mapping)
        )

    transformed_df = transformed_df.fillna(0)

    return transformed_df