import pandas as pd

REQUIRED_COLUMNS = [
    "Age", "Employ", "Address", "Income",
    "Creddebt", "OthDebt", "MonthlyLoad", "Default"
]


def load_raw(path: str) -> pd.DataFrame:
    """
    Carga el dataset crudo desde un CSV.
    """
    df = pd.read_csv(path)
    return df


def validate_schema(df: pd.DataFrame) -> None:
    """
    Lanza ValueError si falta alguna columna requerida.
    """
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]

    if missing:
        raise ValueError(f"Columnas faltantes: {missing}")


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Agrega features derivadas y elimina filas con NaN.
    """
    df = df.copy()

    df["OthDebtRatio"] = df["OthDebt"] / df["Income"]

    df = df.dropna()

    return df

