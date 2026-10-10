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


def iter_cluster_indices(clusters: np.ndarray, n_boot: int, seed: int):
    """Réplicas que remuestrean conglomerados enteros (con reemplazo), no personas."""
    clusters = np.asarray(clusters)
    _, inv = np.unique(clusters, return_inverse=True)
    n_g = int(inv.max()) + 1
    groups = [np.flatnonzero(inv == i) for i in range(n_g)]
    rng = np.random.default_rng(seed)
    for _ in range(n_boot):
        chosen = rng.integers(0, n_g, n_g)
        yield np.concatenate([groups[i] for i in chosen])


def bootstrap_diff_proportions_cluster(
    df: pd.DataFrame,
    y: str,
    group: str,
    cluster: str,
    level_a,
    level_b,
    n_boot: int = 2000,
    seed: int = 42,
    alpha: float = 0.05,
    return_dist: bool = False,
):
    """IC percentil de p_a − p_b remuestreando conglomerados (UPM)."""
    yv = df[y].to_numpy(dtype=float)
    gv = df[group].to_numpy()
    diffs = np.empty(n_boot)
    k = 0
    for idx in iter_cluster_indices(df[cluster].to_numpy(), n_boot, seed):
        yy, gg = yv[idx], gv[idx]
        a = yy[gg == level_a]
        b = yy[gg == level_b]
        diffs[k] = np.nan if (len(a) == 0 or len(b) == 0) else a.mean() - b.mean()
        k += 1
    ok = diffs[np.isfinite(diffs)]
    lo, hi = np.quantile(ok, [alpha / 2, 1 - alpha / 2])
    if return_dist:
        return float(np.nanmean(diffs)), float(lo), float(hi), diffs
    return float(np.nanmean(diffs)), float(lo), float(hi)


def bootstrap_cramers_v_cluster(
    df: pd.DataFrame,
    fila: str,
    col: str,
    cluster: str,
    n_boot: int = 1000,
    seed: int = 42,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """IC percentil de la V de Cramér remuestreando conglomerados."""
    f_codes, _ = pd.factorize(df[fila], sort=False)
    c_codes, _ = pd.factorize(df[col], sort=False)
    r, k = int(f_codes.max()) + 1, int(c_codes.max()) + 1
    vs = np.empty(n_boot)
    for i, idx in enumerate(iter_cluster_indices(df[cluster].to_numpy(), n_boot, seed)):
        tab = np.bincount(f_codes[idx] * k + c_codes[idx], minlength=r * k).reshape(r, k)
        keep_r = tab.sum(axis=1) > 0
        keep_c = tab.sum(axis=0) > 0
        tab = tab[keep_r][:, keep_c]
        if tab.shape[0] < 2 or tab.shape[1] < 2:
            vs[i] = np.nan
        else:
            vs[i] = cramers_v(pd.DataFrame(tab))
    ok = vs[np.isfinite(vs)]
    lo, hi = np.quantile(ok, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def wald_cluster_asociacion(
    df: pd.DataFrame,
    predictor: str,
    outcome: str,
    cluster: str,
) -> tuple[float, int, float]:
    """Wald con sándwich por conglomerado: H0 de que predictor y outcome son independientes.

    Contrasta que los coeficientes de pendiente de las dummies del predictor, en las
    regresiones lineales de las dummies del outcome, son cero. Los gl coinciden con
    los del χ² de independencia.
    """
    y = pd.get_dummies(df[outcome], dtype=float).iloc[:, :-1].to_numpy()
    x = pd.get_dummies(df[predictor], drop_first=True, dtype=float)
    x = np.column_stack([np.ones(len(df)), x.to_numpy()])
    xtx_inv = np.linalg.pinv(x.T @ x)
    b = xtx_inv @ (x.T @ y)
    resid = y - x @ b
    n, k, q = len(df), b.shape[0], b.shape[1]
    # Orden por ecuación: índice = ecuación * k + parámetro, igual que kron(I_q, (X'X)^{-1}).
    scores = np.einsum("nq,nk->nqk", resid, x).reshape(n, q * k)
    _, inv = np.unique(df[cluster].to_numpy(), return_inverse=True)
    g = int(inv.max()) + 1
    meat_g = np.zeros((g, q * k))
    np.add.at(meat_g, inv, scores)
    meat = (meat_g.T @ meat_g) * (g / (g - 1))
    bread = np.kron(np.eye(q), xtx_inv)
    cov = bread @ meat @ bread
    theta_full = np.concatenate([b[:, eq] for eq in range(q)])
    slope = [eq * k + a for eq in range(q) for a in range(1, k)]
    theta = theta_full[slope]
    v = cov[np.ix_(slope, slope)]
    try:
        stat = float(theta @ np.linalg.solve(v, theta))
    except np.linalg.LinAlgError:
        stat = float(theta @ np.linalg.pinv(v) @ theta)
    df_w = len(slope)
    p = float(stats.chi2.sf(stat, df_w))
    return stat, df_w, p
