"""Helpers de modelado predictivo para la Fase 3."""
from __future__ import annotations

from pathlib import Path
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

# Demográficas (Fases 1–2) + económicas (prioridad 1–3)
FEATURES_CAT = [
    "zona",
    "departamento",
    "sexo",
    "condicion_actividad",  # PEAA: ¿trabaja / busca / inactivo?
    "pobreza",  # pobrezai
]
FEATURES_NUM = [
    "edad",
    "log_ipcm",  # log1p(ingreso per cápita del hogar)
]
TARGET = "no_asiste"
FEATURE_COLS = FEATURES_NUM + FEATURES_CAT

MAP_PEAA = {
    "1": "Ocupado",
    "2": "Desocupado",
    "3": "Inactivo",
}
MAP_POBREZA = {
    "1": "Extrema",
    "2": "No extrema",
    "3": "No pobre",
}

# Valores válidos observados en EPHC (sin Chaco / Alto Paraguay / Boquerón).
ZONAS_VALIDAS = ("Urbana", "Rural")
SEXOS_VALIDOS = ("Hombre", "Mujer")
CONDICIONES_ACTIVIDAD = tuple(MAP_PEAA.values())
POBREZAS_VALIDAS = tuple(MAP_POBREZA.values())
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

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_DIR = ROOT / "data" / "raw"
JOIN_KEYS = ["anio_encuesta", "UPM", "NVIVI", "NHOGA", "L02"]


def parse_ipcm(series: pd.Series) -> pd.Series:
    """IPCM viene con decimal coma en los CSV anuales de la EPHC."""
    s = series.astype(str).str.strip().str.replace(",", ".", regex=False)
    s = s.replace({"": np.nan, "nan": np.nan, "None": np.nan})
    return pd.to_numeric(s, errors="coerce")


def load_economic_from_raw(raw_dir: Path | None = None) -> pd.DataFrame:
    """Lee PEAA, pobrezai e ipcm de los REG02 anuales y los estandariza."""
    raw_dir = Path(raw_dir) if raw_dir is not None else DEFAULT_RAW_DIR
    frames: list[pd.DataFrame] = []
    usecols = ["AÑO", "UPM", "NVIVI", "NHOGA", "L02", "PEAA", "pobrezai", "ipcm"]
    for year in (2022, 2023, 2024, 2025):
        path = raw_dir / f"REG02_EPHC_ANUAL_{year}.csv"
        if not path.exists():
            raise FileNotFoundError(f"No está el raw EPHC: {path}")
        raw = pd.read_csv(path, sep=";", usecols=usecols, low_memory=False)
        raw = raw.rename(columns={"AÑO": "anio_encuesta"})
        frames.append(raw)
    eco = pd.concat(frames, ignore_index=True)

    peaa = eco["PEAA"].astype(str).str.strip()
    pob = eco["pobrezai"].astype(str).str.strip()
    eco["condicion_actividad"] = peaa.map(MAP_PEAA)
    eco["pobreza"] = pob.map(MAP_POBREZA)
    eco["log_ipcm"] = np.log1p(parse_ipcm(eco["ipcm"]).clip(lower=0))
    return eco[JOIN_KEYS + ["condicion_actividad", "pobreza", "log_ipcm"]]


def attach_economic_features(
    df: pd.DataFrame,
    raw_dir: Path | None = None,
) -> pd.DataFrame:
    """Une variables económicas al panel Fase 1 vía llaves de hogar/persona."""
    missing_keys = [k for k in JOIN_KEYS if k not in df.columns]
    if missing_keys:
        raise ValueError(
            f"Para enriquecer hacen falta las llaves {missing_keys}. "
            "Usá dataset_fase1.parquet completo (no solo el frame de modelado)."
        )
    if set(["condicion_actividad", "pobreza", "log_ipcm"]).issubset(df.columns):
        return df
    eco = load_economic_from_raw(raw_dir)
    out = df.merge(eco, on=JOIN_KEYS, how="left", validate="one_to_one")
    return out


def load_modeling_frame(path, raw_dir: Path | None = None) -> pd.DataFrame:
    df = pd.read_parquet(path) if str(path).endswith(".parquet") else pd.read_csv(path)
    df = attach_economic_features(df, raw_dir=raw_dir)
    need = FEATURE_COLS + [TARGET, "anio_encuesta"]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas para modelar: {missing}")
    out = df[need].copy()
    out[TARGET] = out[TARGET].astype(int)
    out["edad"] = out["edad"].astype(float)
    out["log_ipcm"] = out["log_ipcm"].astype(float)
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
    return best_threshold_by_fbeta(y_true, y_prob, beta=1.0, grid=grid)


def best_threshold_by_fbeta(
    y_true, y_prob, beta: float = 2.0, grid=None
) -> tuple[float, float]:
    """Elige umbral en validación maximizando F-beta (beta>1 prioriza recall)."""
    if grid is None:
        grid = np.linspace(0.05, 0.95, 37)
    best_t, best_s = 0.5, -1.0
    b2 = beta * beta
    for t in grid:
        pred = (y_prob >= t).astype(int)
        pr = precision_score(y_true, pred, zero_division=0)
        rc = recall_score(y_true, pred, zero_division=0)
        score = ((1 + b2) * pr * rc / (b2 * pr + rc)) if (pr + rc) > 0 else 0.0
        if score > best_s:
            best_s, best_t = float(score), float(t)
    return best_t, best_s


def year_folds(years: list[int]) -> list[tuple[np.ndarray, np.ndarray]]:
    """Validación temporal expanding: entrena con años ≤ t, valida t+1."""
    years = sorted(years)
    folds = []
    for i in range(1, len(years)):
        train_y = years[:i]
        val_y = years[i]
        folds.append((np.array(train_y), np.array([val_y])))
    return folds


def sample_weight_balanced(y) -> np.ndarray:
    """Peso de clase positiva = n_neg / n_pos (para HGB u otros sin class_weight)."""
    y = np.asarray(y)
    pos = (y == 1).sum()
    neg = (y == 0).sum()
    w = np.ones(len(y), dtype=float)
    w[y == 1] = neg / max(pos, 1)
    return w


def fit_temporal_grid(
    estimator,
    param_grid: dict,
    train_df: pd.DataFrame,
    *,
    scale_numeric: bool,
    use_sample_weight: bool = False,
    seed: int = 42,
    feature_cols: list[str] | None = None,
    target: str = TARGET,
) -> tuple[dict, float, pd.DataFrame]:
    """Grid search con validación temporal expanding (solo años de train).

    Para cada combinación de hiperparámetros:
      fold 1: train ≤ 2022 → val 2023
      fold 2: train ≤ 2023 → val 2024
    Elige la combinación con mayor F1 media (umbral provisional 0.5).
    El umbral definitivo se calibra aparte en holdout 2024.
    """
    from sklearn.metrics import f1_score, roc_auc_score
    from sklearn.model_selection import ParameterGrid

    cols = list(feature_cols) if feature_cols is not None else list(FEATURE_COLS)
    years = sorted(train_df["anio_encuesta"].unique())
    folds = year_folds(list(years))
    best_score, best_params, rows = -1.0, None, []

    for params in ParameterGrid(param_grid):
        fold_f1: list[float] = []
        for train_years, val_years in folds:
            tr = train_df[train_df["anio_encuesta"].isin(train_years)]
            va = train_df[train_df["anio_encuesta"].isin(val_years)]
            clf = estimator.__class__(**estimator.get_params())
            clf.set_params(**params)
            if "random_state" in clf.get_params():
                clf.set_params(random_state=seed)
            model = make_pipeline(clf, scale_numeric=scale_numeric)
            fit_kw = {}
            if use_sample_weight:
                fit_kw["clf__sample_weight"] = sample_weight_balanced(tr[target])
            model.fit(tr[cols], tr[target], **fit_kw)
            proba = model.predict_proba(va[cols])[:, 1]
            pred = (proba >= 0.5).astype(int)
            f1 = f1_score(va[target], pred, zero_division=0)
            auc = roc_auc_score(va[target], proba)
            fold_f1.append(float(f1))
            rows.append(
                {
                    "params": str(dict(sorted(params.items()))),
                    "val_year": int(val_years[0]),
                    "f1": float(f1),
                    "auc": float(auc),
                }
            )
        mean_f1 = float(np.mean(fold_f1))
        if mean_f1 > best_score:
            best_score, best_params = mean_f1, dict(params)
    return best_params or {}, best_score, pd.DataFrame(rows)


def make_feature_frame(
    edad: float,
    zona: str,
    departamento: str,
    sexo: str,
    condicion_actividad: str = "Inactivo",
    pobreza: str = "No pobre",
    log_ipcm: float | None = None,
    ipcm: float | None = None,
) -> pd.DataFrame:
    """Arma una fila X con las columnas que esperan los pipelines de Fase 3."""
    edad_f = float(edad)
    if not (12 <= edad_f <= 17):
        raise ValueError(f"edad debe estar entre 12 y 17 (recibido: {edad_f})")
    zona_s = str(zona).strip()
    dep_s = str(departamento).strip()
    sexo_s = str(sexo).strip()
    act_s = str(condicion_actividad).strip()
    pob_s = str(pobreza).strip()
    if zona_s not in ZONAS_VALIDAS:
        raise ValueError(f"zona inválida: {zona_s!r}. Usá una de {ZONAS_VALIDAS}")
    if sexo_s not in SEXOS_VALIDOS:
        raise ValueError(f"sexo inválido: {sexo_s!r}. Usá uno de {SEXOS_VALIDOS}")
    if dep_s not in DEPARTAMENTOS_VALIDOS:
        raise ValueError(
            f"departamento inválido: {dep_s!r}. Usá uno de {DEPARTAMENTOS_VALIDOS}"
        )
    if act_s not in CONDICIONES_ACTIVIDAD:
        raise ValueError(
            f"condicion_actividad inválida: {act_s!r}. Usá una de {CONDICIONES_ACTIVIDAD}"
        )
    if pob_s not in POBREZAS_VALIDAS:
        raise ValueError(f"pobreza inválida: {pob_s!r}. Usá una de {POBREZAS_VALIDAS}")
    if log_ipcm is not None:
        log_v = float(log_ipcm)
    elif ipcm is not None:
        log_v = float(np.log1p(max(float(ipcm), 0.0)))
    else:
        # Mediana aproximada del panel (Gs); solo para demos manuales.
        log_v = float(np.log1p(1_250_000.0))
    return pd.DataFrame(
        [
            {
                "edad": edad_f,
                "log_ipcm": log_v,
                "zona": zona_s,
                "departamento": dep_s,
                "sexo": sexo_s,
                "condicion_actividad": act_s,
                "pobreza": pob_s,
            }
        ]
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
        row0 = X.iloc[0]
        rows.append(
            {
                "modelo": nombre,
                "X_edad": row0["edad"],
                "X_zona": row0["zona"],
                "X_departamento": row0["departamento"],
                "X_sexo": row0["sexo"],
                "X_condicion_actividad": row0["condicion_actividad"],
                "X_pobreza": row0["pobreza"],
                "X_log_ipcm": row0["log_ipcm"],
                "chance_no_asiste": round(proba, 4),
                "umbral": umbral,
                "Y_predicho": pred,
                "etiqueta": "no asiste" if pred == 1 else "asiste",
            }
        )
    return pd.DataFrame(rows)
