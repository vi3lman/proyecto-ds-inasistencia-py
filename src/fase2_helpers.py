"""Helpers estadísticos para la Fase 2 (inferencia)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.proportion import proportions_ztest, confint_proportions_2indep
from statsmodels.stats.multitest import multipletests
import statsmodels.formula.api as smf


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
) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot)
    na, nb = len(x_a), len(x_b)
    for i in range(n_boot):
        a = rng.choice(x_a, size=na, replace=True)
        b = rng.choice(x_b, size=nb, replace=True)
        diffs[i] = a.mean() - b.mean()
    lo, hi = np.quantile(diffs, [alpha / 2, 1 - alpha / 2])
    return float(diffs.mean()), float(lo), float(hi)


def benjamini_hochberg(pvalues: list[float], alpha: float = 0.05):
    reject, p_adj, _, _ = multipletests(pvalues, alpha=alpha, method="fdr_bh")
    return reject, p_adj
