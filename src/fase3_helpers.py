"""Helpers de modelado predictivo para la Fase 3."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


FEATURES_CAT = ["zona", "departamento", "sexo"]
FEATURES_NUM = ["edad"]
TARGET = "no_asiste"
FEATURE_COLS = FEATURES_NUM + FEATURES_CAT


def load_modeling_frame(path) -> pd.DataFrame:
    df = pd.read_parquet(path) if str(path).endswith(".parquet") else pd.read_csv(path)
    need = FEATURE_COLS + [TARGET, "anio_encuesta"]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas para modelar: {missing}")
    out = df[need].copy()
    out[TARGET] = out[TARGET].astype(int)
    out["edad"] = out["edad"].astype(float)
    for c in FEATURES_CAT:
        out[c] = out[c].astype(str)
    return out.dropna(subset=need).reset_index(drop=True)


def chronological_split(df: pd.DataFrame, test_year: int = 2025):
    """Entrena con años anteriores a test_year; prueba solo con test_year."""
    train = df[df["anio_encuesta"] < test_year].copy()
    test = df[df["anio_encuesta"] == test_year].copy()
    if train.empty or test.empty:
        raise ValueError("La partición cronológica quedó vacía.")
    return train, test


def make_preprocessor(scale_numeric: bool = True) -> ColumnTransformer:
    num = StandardScaler() if scale_numeric else "passthrough"
    try:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:  # sklearn < 1.2
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)
    return ColumnTransformer(
        [
            ("num", num, FEATURES_NUM),
            ("cat", ohe, FEATURES_CAT),
        ]
    )


def make_pipeline(estimator, scale_numeric: bool = True) -> Pipeline:
    return Pipeline(
        [
            ("prep", make_preprocessor(scale_numeric=scale_numeric)),
            ("clf", estimator),
        ]
    )


def classification_metrics(y_true, y_pred, y_prob=None) -> dict[str, Any]:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = (int(x) for x in cm.ravel())
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }
    if y_prob is not None and len(np.unique(y_true)) > 1 and np.nanmax(y_prob) > np.nanmin(y_prob):
        out["auc_roc"] = float(roc_auc_score(y_true, y_prob))
        out["auc_pr"] = float(average_precision_score(y_true, y_prob))
    else:
        out["auc_roc"] = float("nan")
        out["auc_pr"] = float("nan")
    return out


def metrics_row(nombre: str, split: str, y_true, y_pred, y_prob=None) -> dict[str, Any]:
    m = classification_metrics(y_true, y_pred, y_prob)
    m["modelo"] = nombre
    m["split"] = split
    return m


def best_threshold_by_f1(y_true, y_prob, grid=None) -> tuple[float, float]:
    """Elige umbral en validación maximizando F1 (no se usa el conjunto de prueba)."""
    if grid is None:
        grid = np.linspace(0.05, 0.95, 19)
    best_t, best_f1 = 0.5, -1.0
    for t in grid:
        f1 = f1_score(y_true, (y_prob >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = float(f1), float(t)
    return best_t, best_f1


def year_folds(years: list[int]) -> list[tuple[np.ndarray, np.ndarray]]:
    """Validación temporal expanding: entrena con años ≤ t, valida t+1."""
    years = sorted(years)
    folds = []
    for i in range(1, len(years)):
        train_y = years[:i]
        val_y = years[i]
        folds.append((np.array(train_y), np.array([val_y])))
    return folds


# Valores válidos observados en EPHC (sin Chaco / Alto Paraguay / Boquerón).
ZONAS_VALIDAS = ("Urbana", "Rural")
SEXOS_VALIDOS = ("Hombre", "Mujer")
DEPARTAMENTOS_VALIDOS = (
    "Alto Paraná",
    "Amambay",
    "Caaguazú",
    "Caazapá",
    "Canindeyú",
    "Capital",
    "Central",
    "Concepción",
    "Cordillera",
    "Guairá",
    "Itapúa",
    "Misiones",
    "Paraguarí",
    "Presidente Hayes",
    "San Pedro",
    "Ñeembucú",
)


def make_feature_frame(
    edad: float,
    zona: str,
    departamento: str,
    sexo: str,
) -> pd.DataFrame:
    """Arma una fila X con las columnas que esperan los pipelines de Fase 3."""
    edad_f = float(edad)
    if not (12 <= edad_f <= 17):
        raise ValueError(f"edad debe estar entre 12 y 17 (recibido: {edad_f})")
    zona_s = str(zona).strip()
    dep_s = str(departamento).strip()
    sexo_s = str(sexo).strip()
    if zona_s not in ZONAS_VALIDAS:
        raise ValueError(f"zona inválida: {zona_s!r}. Usá una de {ZONAS_VALIDAS}")
    if sexo_s not in SEXOS_VALIDOS:
        raise ValueError(f"sexo inválido: {sexo_s!r}. Usá uno de {SEXOS_VALIDOS}")
    if dep_s not in DEPARTAMENTOS_VALIDOS:
        raise ValueError(
            f"departamento inválido: {dep_s!r}. Usá uno de {DEPARTAMENTOS_VALIDOS}"
        )
    return pd.DataFrame(
        [{"edad": edad_f, "zona": zona_s, "departamento": dep_s, "sexo": sexo_s}]
    )[FEATURE_COLS]


def predict_with_models(
    X: pd.DataFrame,
    fitted: dict[str, Any],
    thresholds: dict[str, float],
) -> pd.DataFrame:
    """Para cada modelo: probabilidad de no asistir, decisión 0/1 y etiqueta."""
    rows: list[dict[str, Any]] = []
    for nombre, model in fitted.items():
        umbral = float(thresholds.get(nombre, 0.5))
        if hasattr(model, "predict_proba"):
            proba = float(model.predict_proba(X)[0, 1])
        else:
            pred0 = int(model.predict(X)[0])
            proba = float(pred0)
        pred = int(proba >= umbral)
        rows.append(
            {
                "modelo": nombre,
                "X_edad": X.iloc[0]["edad"],
                "X_zona": X.iloc[0]["zona"],
                "X_departamento": X.iloc[0]["departamento"],
                "X_sexo": X.iloc[0]["sexo"],
                "chance_no_asiste": round(proba, 4),
                "umbral": umbral,
                "Y_predicho": pred,
                "etiqueta": "no asiste" if pred == 1 else "asiste",
            }
        )
    return pd.DataFrame(rows)
