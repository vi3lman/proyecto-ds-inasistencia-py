"""Helpers estadísticos para la Fase 2 (inferencia)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests


MOTIVO_GRUPO = {
    "Sin recursos en el hogar": "Económico",
    "Necesidad de trabajar": "Económico",
    "Muy costosos materiales y matrículas": "Económico",
    "Motivos familiares": "Familiar",
    "Debe hacer labores en el hogar": "Familiar",
    "No existe institución cercana": "Oferta",
    "Institución cercana muy mala": "Oferta",
    "El centro educativo cerró": "Oferta",
    "El docente no asiste con regularidad": "Oferta",
    "Institución no ofrece escolaridad completa": "Oferta",
    "No quiere estudiar": "Personal",
    "Considera que terminó los estudios": "Personal",
    "Asiste a formación profesional/vocacional": "Personal",
    "Por enfermedad": "Otro",
    "Requiere educación especial": "Otro",
    "Otra razón": "Otro",
}


def cramers_v(table: pd.DataFrame) -> float:
    chi2 = stats.chi2_contingency(table, correction=False)[0]
    n = table.to_numpy().sum()
    r, k = table.shape
    return float(np.sqrt(chi2 / (n * (min(r, k) - 1))))


def cohen_h(p1: float, p2: float) -> float:
    return float(2 * (np.arcsin(np.sqrt(p1)) - np.arcsin(np.sqrt(p2))))


def bootstrap_diff_proportions(
    x_a: np.ndarray,
    x_b: np.ndarray,
    n_boot: int = 2000,
    seed: int = 42,
    alpha: float = 0.05,
    return_dist: bool = False,
):
    """IC percentil de p_a − p_b. Con return_dist=True devuelve también las réplicas."""
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot)
    na, nb = len(x_a), len(x_b)
    for i in range(n_boot):
        a = rng.choice(x_a, size=na, replace=True)
        b = rng.choice(x_b, size=nb, replace=True)
        diffs[i] = a.mean() - b.mean()
    lo, hi = np.quantile(diffs, [alpha / 2, 1 - alpha / 2])
    if return_dist:
        return float(diffs.mean()), float(lo), float(hi), diffs
    return float(diffs.mean()), float(lo), float(hi)


def benjamini_hochberg(pvalues: list[float], alpha: float = 0.05):
    reject, p_adj, _, _ = multipletests(pvalues, alpha=alpha, method="fdr_bh")
    return reject, p_adj


def chi2_supuestos(expected: np.ndarray) -> dict:
    """Supuesto de Cochran para χ²: mínimo esperado y % de celdas esperadas < 5."""
    expected = np.asarray(expected)
    return {
        "min_esperado": float(expected.min()),
        "pct_celdas_menor_5": float(100 * (expected < 5).mean()),
        "cumple": bool(expected.min() >= 1 and (expected < 5).mean() <= 0.20),
    }


def residuos_ajustados(table: pd.DataFrame) -> pd.DataFrame:
    """Residuos estandarizados ajustados de Haberman: ~N(0,1) bajo independencia."""
    obs = table.to_numpy(dtype=float)
    n = obs.sum()
    fila = obs.sum(axis=1, keepdims=True) / n
    col = obs.sum(axis=0, keepdims=True) / n
    esp = n * fila * col
    res = (obs - esp) / np.sqrt(esp * (1 - fila) * (1 - col))
    return pd.DataFrame(res, index=table.index, columns=table.columns)


def bootstrap_cramers_v(
    df: pd.DataFrame,
    fila: str,
    col: str,
    n_boot: int = 1000,
    seed: int = 42,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """IC percentil de la V de Cramér remuestreando filas (personas) con reemplazo."""
    rng = np.random.default_rng(seed)
    f_codes, f_cats = pd.factorize(df[fila])
    c_codes, c_cats = pd.factorize(df[col])
    r, k, n = len(f_cats), len(c_cats), len(df)
    vs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        tab = np.bincount(f_codes[idx] * k + c_codes[idx], minlength=r * k).reshape(r, k)
        tab = tab[tab.sum(axis=1) > 0][:, tab.sum(axis=0) > 0]
        vs[i] = cramers_v(pd.DataFrame(tab))
    lo, hi = np.quantile(vs, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def spearman_ci(rho: float, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """IC de ρ de Spearman por transformación z de Fisher (EE de Bonett–Wright)."""
    se = np.sqrt((1 + rho**2 / 2) / (n - 3))
    z = np.arctanh(rho)
    q = stats.norm.ppf(1 - alpha / 2)
    return float(np.tanh(z - q * se)), float(np.tanh(z + q * se))
