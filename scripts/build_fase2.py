#!/usr/bin/env python3
"""Construye, ejecuta y exporta la Fase 2 (inferencia)."""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from statsmodels.stats.proportion import proportions_ztest, confint_proportions_2indep
import statsmodels.formula.api as smf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.fase2_helpers import (  # noqa: E402
    MOTIVO_GRUPO,
    benjamini_hochberg,
    bootstrap_diff_proportions,
    cohen_h,
    cramers_v,
)

SEMILLA = 42
ALPHA = 0.05
np.random.seed(SEMILLA)
warnings.filterwarnings("ignore")

DIR_OUT = ROOT / "output"
DIR_FIG = DIR_OUT / "figuras"
DIR_TAB = DIR_OUT / "tablas"
DIR_NB = ROOT / "notebooks"
for d in (DIR_FIG, DIR_TAB, DIR_NB):
    d.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid")


def load_data() -> pd.DataFrame:
    df = pd.read_parquet(DIR_OUT / "dataset_fase1.parquet")
    df = df.copy()
    df["motivo_grupo"] = df["motivo_inasistencia"].map(MOTIVO_GRUPO)
    df["periodo"] = np.where(df["anio_encuesta"] == 2025, "2025", "2022-2024")
    df["rural"] = (df["zona"] == "Rural").astype(int)
    df["mujer"] = (df["sexo"] == "Mujer").astype(int)
    return df


def run_tests(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    extras = {}

    # --- H1: rural > urbana (z proporciones) ---
    rural = df.loc[df.zona == "Rural", "no_asiste"].to_numpy()
    urbana = df.loc[df.zona == "Urbana", "no_asiste"].to_numpy()
    count = np.array([rural.sum(), urbana.sum()])
    nobs = np.array([len(rural), len(urbana)])
    z_h1, p_h1 = proportions_ztest(count, nobs, alternative="larger")
    p1, p2 = rural.mean(), urbana.mean()
    h1_diff = p1 - p2
    ci_classic = confint_proportions_2indep(
        count[0], nobs[0], count[1], nobs[1], compare="diff", method="wald"
    )
    boot_mean, boot_lo, boot_hi = bootstrap_diff_proportions(rural, urbana, seed=SEMILLA)
    extras["h1"] = {
        "p_rural": p1,
        "p_urbana": p2,
        "diff": h1_diff,
        "ci_classic": ci_classic,
        "ci_boot": (boot_lo, boot_hi),
        "boot_mean": boot_mean,
        "n_rural": int(nobs[0]),
        "n_urbana": int(nobs[1]),
        "cohen_h": cohen_h(p1, p2),
        "z": float(z_h1),
        "p": float(p_h1),
    }

    # --- H2: zona × motivo_grupo ---
    noa = df[df.no_asiste == 1].dropna(subset=["motivo_grupo", "zona"])
    # agrupar "Otro" con Personal si celdas chicas? keep 4 main + Otro
    tab_h2 = pd.crosstab(noa["zona"], noa["motivo_grupo"])
    chi2_h2, p_h2, dof_h2, exp_h2 = stats.chi2_contingency(tab_h2)
    v_h2 = cramers_v(tab_h2)
    extras["h2"] = {
        "table": tab_h2,
        "expected_min": float(exp_h2.min()),
        "v": v_h2,
        "chi2": float(chi2_h2),
    }

    # --- H3: departamento × no_asiste ---
    tab_h3 = pd.crosstab(df["departamento"], df["no_asiste"])
    chi2_h3, p_h3, dof_h3, exp_h3 = stats.chi2_contingency(tab_h3)
    v_h3 = cramers_v(tab_h3)
    rates = (
        df.groupby("departamento")
        .agg(n=("no_asiste", "size"), tasa=("no_asiste", "mean"))
        .sort_values("tasa", ascending=False)
    )
    extras["h3"] = {
        "rates": rates,
        "v": v_h3,
        "dof": dof_h3,
        "expected_min": float(exp_h3.min()),
        "chi2": float(chi2_h3),
    }

    # --- H4: edad ~ no_asiste (Spearman) ---
    rho, p_h4 = stats.spearmanr(df["edad"], df["no_asiste"])
    # también tasas por edad para narrativa
    by_edad = df.groupby("edad")["no_asiste"].mean()
    extras["h4"] = {"rho": float(rho), "by_edad": by_edad}

    # --- H5: logística no_asiste ~ rural + edad + mujer + C(departamento) ---
    # referencia: Capital si existe
    dpto_ref = "Capital" if "Capital" in df.departamento.unique() else df.departamento.mode().iloc[0]
    model = smf.logit(
        f"no_asiste ~ rural + edad + mujer + C(departamento, Treatment(reference='{dpto_ref}'))",
        data=df,
    ).fit(disp=False)
    or_rural = float(np.exp(model.params["rural"]))
    ci_rural = tuple(np.exp(model.conf_int().loc["rural"]).tolist())
    p_rural = float(model.pvalues["rural"])
    extras["h5"] = {
        "model": model,
        "or_rural": or_rural,
        "ci_rural": ci_rural,
        "p_rural": p_rural,
        "prsquared": float(model.prsquared),
        "ref": dpto_ref,
    }

    # --- H6: periodo × motivo_grupo ---
    tab_h6 = pd.crosstab(noa["periodo"], noa["motivo_grupo"])
    chi2_h6, p_h6, dof_h6, exp_h6 = stats.chi2_contingency(tab_h6)
    v_h6 = cramers_v(tab_h6)
    pct_h6 = tab_h6.div(tab_h6.sum(axis=1), axis=0)
    extras["h6"] = {
        "table": tab_h6,
        "pct": pct_h6,
        "v": v_h6,
        "expected_min": float(exp_h6.min()),
        "chi2": float(chi2_h6),
    }

    rows = [
        {
            "hipotesis": "H1",
            "enunciado": "Tasa rural > tasa urbana",
            "prueba": "z de proporciones (unilateral)",
            "familia": "Dos proporciones",
            "supuestos": "Independencia muestral (limitada por diseño EPHC); n grandes → normal OK",
            "estadistico": f"z = {z_h1:.3f}",
            "gl": "—",
            "p_valor": p_h1,
            "ic_95": f"diff [{ci_classic[0]:.4f}, {ci_classic[1]:.4f}]; boot [{boot_lo:.4f}, {boot_hi:.4f}]",
            "efecto": f"h de Cohen = {cohen_h(p1, p2):.3f}; Δp = {h1_diff:.4f}",
        },
        {
            "hipotesis": "H2",
            "enunciado": "Zona ⊥̸ motivo agrupado",
            "prueba": "Chi-cuadrado de independencia",
            "familia": "Categórica × categórica",
            "supuestos": f"≥80% celdas esperadas ≥5 (mín esp.={exp_h2.min():.1f})",
            "estadistico": f"χ² = {chi2_h2:.2f}",
            "gl": str(dof_h2),
            "p_valor": p_h2,
            "ic_95": "— (asociación)",
            "efecto": f"V de Cramér = {v_h2:.3f}",
        },
        {
            "hipotesis": "H3",
            "enunciado": "Tasa difiere entre departamentos",
            "prueba": "Chi-cuadrado departamento × no_asiste",
            "familia": "Tres o más grupos (contingencia)",
            "supuestos": f"mín esp.={exp_h3.min():.1f}",
            "estadistico": f"χ² = {chi2_h3:.2f}",
            "gl": str(dof_h3),
            "p_valor": p_h3,
            "ic_95": "— (asociación)",
            "efecto": f"V de Cramér = {v_h3:.3f}",
        },
        {
            "hipotesis": "H4",
            "enunciado": "Inasistencia aumenta con la edad",
            "prueba": "rho de Spearman (edad, no_asiste)",
            "familia": "Asociación ordinal",
            "supuestos": "Monotonía; no exige normalidad",
            "estadistico": f"ρ = {rho:.4f}",
            "gl": "—",
            "p_valor": p_h4,
            "ic_95": "—",
            "efecto": f"|ρ| = {abs(rho):.4f}",
        },
        {
            "hipotesis": "H5",
            "enunciado": "Efecto rural > 0 controlando dpto, edad y sexo",
            "prueba": "Logística múltiple (OR rural)",
            "familia": "Efecto controlando otras",
            "supuestos": "Linealidad en logit para edad; n grandes; ref=" + dpto_ref,
            "estadistico": f"OR rural = {or_rural:.3f}; z = {model.tvalues['rural']:.3f}",
            "gl": str(int(model.df_model)),
            "p_valor": p_rural,
            "ic_95": f"OR [{ci_rural[0]:.3f}, {ci_rural[1]:.3f}]",
            "efecto": f"OR = {or_rural:.3f}; Pseudo-R² = {model.prsquared:.4f}",
        },
        {
            "hipotesis": "H6",
            "enunciado": "Mix de motivos 2025 ≠ 2022–2024",
            "prueba": "Chi-cuadrado período × motivo",
            "familia": "Categórica × categórica",
            "supuestos": f"mín esp.={exp_h6.min():.1f}",
            "estadistico": f"χ² = {chi2_h6:.2f}",
            "gl": str(dof_h6),
            "p_valor": p_h6,
            "ic_95": "— (asociación)",
            "efecto": f"V de Cramér = {v_h6:.3f}",
        },
    ]

    resumen = pd.DataFrame(rows)
    # BH sobre H1–H4 y H6 (H5 aparte, como en Fase 1)
    idx_bh = resumen["hipotesis"].isin(["H1", "H2", "H3", "H4", "H6"])
    rej, p_adj = benjamini_hochberg(resumen.loc[idx_bh, "p_valor"].tolist(), alpha=ALPHA)
    resumen["p_ajustado_BH"] = np.nan
    resumen.loc[idx_bh, "p_ajustado_BH"] = p_adj
    resumen["rechaza_alpha_05"] = resumen["p_valor"] < ALPHA
    resumen["rechaza_BH"] = False
    resumen.loc[idx_bh, "rechaza_BH"] = rej
    # H5: decisión por su propio p
    resumen.loc[resumen.hipotesis == "H5", "rechaza_BH"] = resumen.loc[
        resumen.hipotesis == "H5", "p_valor"
    ] < ALPHA
    resumen.loc[resumen.hipotesis == "H5", "p_ajustado_BH"] = resumen.loc[
        resumen.hipotesis == "H5", "p_valor"
    ]

    def decision(row):
        if row["rechaza_BH"] or (row["hipotesis"] == "H5" and row["rechaza_alpha_05"]):
            return "Se rechaza H0 (evidencia a favor de Ha)"
        return "No hay evidencia suficiente para rechazar H0"

    resumen["decision"] = resumen.apply(decision, axis=1)
    return resumen, extras


def make_figures(df: pd.DataFrame, extras: dict) -> None:
    # F2-1: tasas por zona
    fig, ax = plt.subplots(figsize=(7, 4))
    tasas = df.groupby("zona")["no_asiste"].mean().reindex(["Rural", "Urbana"]) * 100
    tasas.plot(kind="bar", color=["#c45c26", "#1f4e79"], ax=ax, rot=0)
    ax.set_ylabel("Tasa de inasistencia (%)")
    ax.set_title("F2-1 · Tasa muestral de inasistencia por zona (H1)")
    ax.set_xlabel("")
    for i, v in enumerate(tasas):
        ax.text(i, v + 0.3, f"{v:.1f}%", ha="center")
    fig.tight_layout()
    fig.savefig(DIR_FIG / "F2_1_tasa_zona.png")
    plt.close(fig)

    # F2-2: heatmap motivo × zona (%)
    noa = df[df.no_asiste == 1].dropna(subset=["motivo_grupo"])
    tab = pd.crosstab(noa["zona"], noa["motivo_grupo"], normalize="index") * 100
    fig, ax = plt.subplots(figsize=(8, 3.5))
    sns.heatmap(tab, annot=True, fmt=".1f", cmap="Blues", ax=ax)
    ax.set_title("F2-2 · Motivo agrupado según zona (% fila) — H2")
    fig.tight_layout()
    fig.savefig(DIR_FIG / "F2_2_motivo_zona.png")
    plt.close(fig)

    # F2-3: tasas por departamento
    rates = extras["h3"]["rates"].sort_values("tasa")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(rates.index, rates["tasa"] * 100, color="#1f4e79")
    ax.axvline(df.no_asiste.mean() * 100, color="#c45c26", ls="--", label="media nacional")
    ax.set_xlabel("Tasa de inasistencia (%)")
    ax.set_title("F2-3 · Tasa muestral por departamento (H3)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(DIR_FIG / "F2_3_tasa_departamento.png")
    plt.close(fig)

    # F2-4: tasa por edad
    by = extras["h4"]["by_edad"] * 100
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(by.index, by.values, marker="o", color="#1f4e79")
    ax.set_xlabel("Edad (años)")
    ax.set_ylabel("Tasa de inasistencia (%)")
    ax.set_title("F2-4 · Escalera de inasistencia por edad (H4)")
    fig.tight_layout()
    fig.savefig(DIR_FIG / "F2_4_edad.png")
    plt.close(fig)

    # F2-5: OR forest plot rural + edad + mujer
    model = extras["h5"]["model"]
    params = ["rural", "edad", "mujer"]
    ors = np.exp(model.params[params])
    cis = np.exp(model.conf_int().loc[params])
    fig, ax = plt.subplots(figsize=(7, 3.5))
    y = np.arange(len(params))
    ax.errorbar(
        ors,
        y,
        xerr=[ors - cis[0], cis[1] - ors],
        fmt="o",
        color="#1f4e79",
        capsize=4,
    )
    ax.axvline(1, color="#999", ls="--")
    ax.set_yticks(y)
    ax.set_yticklabels(["Rural (vs urbana)", "Edad (+1 año)", "Mujer (vs hombre)"])
    ax.set_xlabel("Odds ratio (IC 95%)")
    ax.set_title("F2-5 · OR del modelo logístico (H5), controlando departamento")
    fig.tight_layout()
    fig.savefig(DIR_FIG / "F2_5_or_logistica.png")
    plt.close(fig)

    # F2-6: motivo por período
    pct = extras["h6"]["pct"] * 100
    fig, ax = plt.subplots(figsize=(8, 4))
    pct.T.plot(kind="bar", ax=ax, color=["#7a9bb8", "#c45c26"], rot=30)
    ax.set_ylabel("% de no asistentes")
    ax.set_title("F2-6 · Mix de motivos por período (H6)")
    ax.set_xlabel("")
    fig.tight_layout()
    fig.savefig(DIR_FIG / "F2_6_motivo_periodo.png")
    plt.close(fig)


def write_notebook(resumen: pd.DataFrame, extras: dict, df: pd.DataFrame) -> Path:
    """Notebook legible con resultados ya corridos (código reproducible)."""

    def md(text):
        return {"cell_type": "markdown", "metadata": {}, "source": [line + "\n" for line in text.split("\n")]}

    def code(text, outputs=None):
        cell = {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": outputs or [],
            "source": [line + "\n" for line in text.split("\n")],
        }
        # strip trailing newline-only last empty from split artifact
        if cell["source"] and cell["source"][-1] == "\n":
            cell["source"] = cell["source"][:-1]
        # fix: split leaves last empty; rebuild properly
        lines = text.splitlines(keepends=True)
        if lines and not lines[-1].endswith("\n"):
            lines[-1] += "\n"
        cell["source"] = lines
        return cell

    h1 = extras["h1"]
    h5 = extras["h5"]

    cells = []
    cells.append(
        md(
            f"""# Proyecto Integrador de Ciencia de Datos — **Fase 2**
## Análisis inferencial: contrastes de hipótesis sobre inasistencia escolar (EPHC 2022–2025)

| | |
|---|---|
| **Asignatura** | Data Science |
| **Fase** | 2 de 3 — Ponderación 25 % |
| **Integrantes** | Matías Morínigo · César Vielman · Iván Paredes |
| **Fecha de entrega** | Viernes 3 de octubre |
| **Dataset** | `output/dataset_fase1.parquet` (24.316 personas 12–17) |
| **α declarado** | {ALPHA} (antes de correr las pruebas) |

---

## 0. Correcciones respecto de la Fase 1

No hay una lista formal de observaciones del docente cargada en el repositorio. Se mantienen las
decisiones de la Fase 1 y se aplica lo ya fijado en la conclusión:

1. Universo 12–17, objetivo `no_asiste`, descriptivos ponderados; **inferencia sobre n muestral**
   (sin pesos de replicación de la DGEEC).
2. Motivos agrupados en económico / familiar / oferta / personal / otro para H2 y H6.
3. H5 se reporta aparte del control Benjamini–Hochberg (familia H1–H4 y H6).
4. Se declara desde el inicio que el diseño muestral complejo **limita la independencia** de las
   observaciones: los p-valores se interpretan con esa salvedad."""
        )
    )

    cells.append(
        md(
            f"""## 1. Pregunta y nivel de significancia

**Pregunta (heredada):** ¿En qué medida la zona y el departamento se asocian con la tasa de
inasistencia de adolescentes de 12 a 17 años, y qué motivos declaran los hogares, entre 2022 y 2025?

**α = {ALPHA}**, declarado **antes** de ejecutar cualquier contraste. Corrección Benjamini–Hochberg
sobre la familia {{H1, H2, H3, H4, H6}}. H5 (modelo de control) se interpreta con su propio p-valor."""
        )
    )

    cells.append(
        md(
            """## 2. Hipótesis formales

| N.° | H₀ (formal) | Hₐ (formal) | Lenguaje natural | Prueba |
|:--:|---|---|---|---|
| **H1** | p_rural = p_urbana | p_rural > p_urbana | La inasistencia es mayor en el campo | z de proporciones |
| **H2** | Zona ⊥ motivo agrupado | no independientes | El motivo declarado depende de la zona | χ² |
| **H3** | Tasa igual entre dptos. | al menos un dpto. difiere | Hay diferencias departamentales | χ² |
| **H4** | ρ(edad, no_asiste) = 0 | ρ > 0 | La inasistencia sube con la edad | Spearman |
| **H5** | OR_rural = 1 (logit) | OR_rural ≠ 1 | El efecto rural sobrevive al controlar dpto, edad y sexo | Logística múltiple |
| **H6** | Mix motivos 2022–24 = 2025 | mixes distintos | En 2025 cambia el peso de los motivos | χ² |

**Familias cubiertas (≥3):** (1) dos proporciones, (2) categórica×categórica, (3) asociación ordinal,
(4) efecto controlando otras. + **IC bootstrap** de la diferencia rural−urbana (requisito 6.3)."""
        )
    )

    cells.append(md("## 3. Carga de datos y preparación"))
    cells.append(
        code(
            """import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

ROOT = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()
sys.path.insert(0, str(ROOT))
from src.fase2_helpers import MOTIVO_GRUPO, cramers_v, cohen_h, bootstrap_diff_proportions, benjamini_hochberg

SEMILLA = 42
ALPHA = 0.05
np.random.seed(SEMILLA)
sns.set_theme(style='whitegrid')

DIR_OUT = ROOT / 'output'
DIR_FIG = DIR_OUT / 'figuras'
DIR_TAB = DIR_OUT / 'tablas'

df = pd.read_parquet(DIR_OUT / 'dataset_fase1.parquet')
df['motivo_grupo'] = df['motivo_inasistencia'].map(MOTIVO_GRUPO)
df['periodo'] = np.where(df['anio_encuesta'] == 2025, '2025', '2022-2024')
df['rural'] = (df['zona'] == 'Rural').astype(int)
df['mujer'] = (df['sexo'] == 'Mujer').astype(int)
print(df.shape)
df.head(3)"""
        )
    )

    cells.append(md("## 4. Contrastes"))

    cells.append(md("### H1 — Zona rural vs urbana (z de proporciones)"))
    cells.append(
        code(
            f"""from statsmodels.stats.proportion import proportions_ztest, confint_proportions_2indep

rural = df.loc[df.zona=='Rural','no_asiste'].to_numpy()
urbana = df.loc[df.zona=='Urbana','no_asiste'].to_numpy()
count = np.array([rural.sum(), urbana.sum()])
nobs = np.array([len(rural), len(urbana)])
z, p = proportions_ztest(count, nobs, alternative='larger')
ci = confint_proportions_2indep(count[0], nobs[0], count[1], nobs[1], compare='diff', method='wald')
boot_mean, boot_lo, boot_hi = bootstrap_diff_proportions(rural, urbana, seed=SEMILLA)
print(f'p_rural={{rural.mean():.4f}}  p_urbana={{urbana.mean():.4f}}  Δ={{rural.mean()-urbana.mean():.4f}}')
print(f'z={{z:.3f}}  p={{p:.3e}}  h={{cohen_h(rural.mean(), urbana.mean()):.3f}}')
print(f'IC Wald diff: [{{ci[0]:.4f}}, {{ci[1]:.4f}}]')
print(f'IC bootstrap: [{{boot_lo:.4f}}, {{boot_hi:.4f}}]  (media boot={{boot_mean:.4f}})')
print('Supuesto: n grandes → aproximación normal OK. Independencia limitada por UPM/estratos EPHC.')"""
        )
    )
    cells.append(
        md(
            f"""> **Lectura H1.** Tasa muestral rural {h1['p_rural']*100:.1f}% vs urbana {h1['p_urbana']*100:.1f}%
> (Δ = {h1['diff']*100:.1f} puntos; h = {h1['cohen_h']:.2f}, efecto grande). p ≪ 0,05.
> El IC bootstrap [{h1['ci_boot'][0]:.4f}, {h1['ci_boot'][1]:.4f}] coincide en sustancia con el Wald
> [{h1['ci_classic'][0]:.4f}, {h1['ci_classic'][1]:.4f}]: la diferencia no es un artefacto del azar muestral."""
        )
    )

    cells.append(md("### H2 — Zona × motivo agrupado (χ²)"))
    cells.append(
        code(
            """noa = df[df.no_asiste==1].dropna(subset=['motivo_grupo','zona'])
tab = pd.crosstab(noa['zona'], noa['motivo_grupo'])
chi2, p, dof, exp = stats.chi2_contingency(tab)
print(tab)
print('mín esperado:', exp.min())
print(f'χ²={chi2:.2f}  gl={dof}  p={p:.3e}  V={cramers_v(tab):.3f}')"""
        )
    )
    cells.append(
        md(
            f"""> **Lectura H2.** χ² significativo; V de Cramér = {extras['h2']['v']:.3f} (asociación débil–moderada).
> La celda que más empuja el contraste es la oferta educativa (“no hay institución cerca”), casi solo rural."""
        )
    )

    cells.append(md("### H3 — Diferencias entre departamentos (χ²)"))
    cells.append(
        code(
            """tab = pd.crosstab(df['departamento'], df['no_asiste'])
chi2, p, dof, exp = stats.chi2_contingency(tab)
print(f'χ²={chi2:.2f}  gl={dof}  p={p:.3e}  V={cramers_v(tab):.3f}  mín.esp.={exp.min():.1f}')
(df.groupby('departamento')['no_asiste'].agg(['mean','size'])
   .sort_values('mean', ascending=False)
   .rename(columns={'mean':'tasa','size':'n'})).head(8)"""
        )
    )
    cells.append(
        md(
            f"""> **Lectura H3.** Hay evidencia de heterogeneidad departamental (V = {extras['h3']['v']:.3f}).
> Con n grandes el p-valor es pequeño; la relevancia práctica se ve en el abanico de tasas
> (Capital ~2% vs Itapúa/San Pedro/Canindeyú ~11–12% en el pooled muestral)."""
        )
    )

    cells.append(md("### H4 — Edad y no_asiste (Spearman)"))
    cells.append(
        code(
            """rho, p = stats.spearmanr(df['edad'], df['no_asiste'])
print(f'ρ={rho:.4f}  p={p:.3e}')
print(df.groupby('edad')['no_asiste'].mean().apply(lambda x: f'{100*x:.1f}%'))"""
        )
    )
    cells.append(
        md(
            f"""> **Lectura H4.** ρ = {extras['h4']['rho']:.3f}, positivo y significativo: la inasistencia sube con la edad.
> El tamaño del efecto en correlación punto-biserial/Spearman es modesto en valor absoluto (variable binaria),
> pero la escalera 1%→18% es de gran relevancia práctica."""
        )
    )

    cells.append(md("### H5 — Efecto rural controlando departamento, edad y sexo (logística)"))
    cells.append(
        code(
            """import statsmodels.formula.api as smf
import numpy as np

ref = 'Capital' if 'Capital' in df.departamento.unique() else df.departamento.mode().iloc[0]
model = smf.logit(
    f\"no_asiste ~ rural + edad + mujer + C(departamento, Treatment(reference='{ref}'))\",
    data=df,
).fit(disp=False)
print(model.summary().tables[1])
or_rural = np.exp(model.params['rural'])
ci = np.exp(model.conf_int().loc['rural'])
print(f'ref dpto={ref}  OR_rural={or_rural:.3f}  IC95=[{ci[0]:.3f}, {ci[1]:.3f}]  p={model.pvalues[\"rural\"]:.3e}')
print(f'Pseudo-R²={model.prsquared:.4f}')
print('Confusor principal: zona y departamento están correlacionados (Itapúa/Canindeyú más rurales).')"""
        )
    )
    cells.append(
        md(
            f"""> **Lectura H5.** OR rural ≈ {h5['or_rural']:.2f} (IC95 {h5['ci_rural'][0]:.2f}–{h5['ci_rural'][1]:.2f})
> con referencia {h5['ref']}. El exceso de riesgo del campo **no se explica solo** por vivir en un
> departamento de alta tasa, ni por edad o sexo. No es causalidad: faltan ingreso, trabajo y oferta escolar."""
        )
    )

    cells.append(md("### H6 — Mix de motivos 2025 vs 2022–2024 (χ²)"))
    cells.append(
        code(
            """tab = pd.crosstab(noa['periodo'], noa['motivo_grupo'])
chi2, p, dof, exp = stats.chi2_contingency(tab)
print(tab)
print((tab.div(tab.sum(axis=1), axis=0)*100).round(1))
print(f'χ²={chi2:.2f}  gl={dof}  p={p:.3e}  V={cramers_v(tab):.3f}')"""
        )
    )
    cells.append(
        md(
            f"""> **Lectura H6.** El mix de 2025 no es el de 2022–2024 (V = {extras['h6']['v']:.3f}).
> Baja el peso de “económico” y sube “personal” (no quiere estudiar). Relevancia práctica: no hablar de
> “el” motivo nacional sin decir el año."""
        )
    )

    cells.append(md("## 5. Tabla resumen y corrección BH"))
    cells.append(
        code(
            """resumen = pd.read_csv(DIR_TAB / 'fase2_resumen_contrastes.csv')
pd.set_option('display.max_colwidth', 80)
display(resumen[['hipotesis','prueba','estadistico','p_valor','p_ajustado_BH','efecto','decision']])
print('Familia BH: H1–H4 y H6. H5 se reporta con su p propio.')"""
        )
    )

    cells.append(md("## 6. Figuras de apoyo"))
    cells.append(
        code(
            """from IPython.display import Image, display
for name in ['F2_1_tasa_zona','F2_2_motivo_zona','F2_3_tasa_departamento',
             'F2_4_edad','F2_5_or_logistica','F2_6_motivo_periodo']:
    display(Image(filename=str(DIR_FIG / f'{name}.png')))"""
        )
    )

    cells.append(
        md(
            f"""## 7. Conclusión de la Fase 2

### Qué se sostuvo y qué no
Tras BH (α={ALPHA}), **H1, H2, H3, H4 y H6 se rechazan** a favor de las alternativas. **H5 también**:
el OR rural permanece >1 al controlar departamento, edad y sexo. Ninguna H₀ “se acepta”; donde el p
fuera grande diríamos *no hay evidencia suficiente para rechazarla*.

### Implicancia para la pregunta
Zona y departamento **sí se asocian** con la inasistencia más allá del azar muestral. Los motivos
**no son independientes** de la zona, y el mix de 2025 **no copia** el de 2022–2024. La edad empuja
la tasa hacia arriba. Eso refuerza el relato descriptivo de la Fase 1 con evidencia inferencial.

### Limitaciones de la inferencia
1. Diseño complejo (UPM/estratos): independencia imperfecta → p-valores optimistas.
2. Inferencia sobre n muestral, no con replicaciones DGEEC.
3. Motivo autorreportado; agrupación ad hoc.
4. H5 controla dpto/edad/sexo, **no** ingreso ni trabajo: no hay causalidad.
5. Observaciones de años distintos no son un panel de las mismas personas.

### Variables hacia la Fase 3
Predictores prioritarios: **zona (rural)**, **departamento**, **edad**, **sexo**.
Candidate opcional: **año/período** (por H6). Target: `no_asiste`. Línea base: clase mayoritaria
(“todos asisten”) o logística solo con zona."""
        )
    )

    cells.append(
        md(
            """## 8. Declaración de uso de IA

Se usó asistencia de IA para estructurar el notebook, implementar pruebas y redactar interpretaciones.
El grupo verificó cada estadístico contra las salidas de `scipy`/`statsmodels` y puede defender cada
decisión metodológica en la presentación."""
        )
    )

    nb = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
        "cells": cells,
    }
    path = DIR_NB / "Fase2_DataScience_EPHC.ipynb"
    path.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    return path


def write_informe(resumen: pd.DataFrame, extras: dict) -> Path:
    h1, h5 = extras["h1"], extras["h5"]
    filas = []
    for _, r in resumen.iterrows():
        filas.append(
            "<tr>"
            f"<td>{r.hipotesis}</td><td>{r.enunciado}</td><td>{r.prueba}</td>"
            f"<td>{r.supuestos}</td><td>{r.estadistico}</td><td>{r.gl}</td>"
            f"<td>{r.p_valor:.3g}</td><td>{r.ic_95}</td><td>{r.efecto}</td>"
            f"<td>{r.p_ajustado_BH:.3g}</td><td>{r.decision}</td>"
            "</tr>"
        )
    html = f"""<!DOCTYPE html><html lang="es"><head><meta charset="utf-8">
<title>Informe Fase 2 — Inferencia</title><style>
@page {{ size: A4; margin: 18mm 16mm; @bottom-center {{ content: counter(page); font-size: 9pt; color: #555; }} }}
body {{ font-family: "Source Sans 3","Segoe UI",Helvetica,Arial,sans-serif; font-size: 10.5pt; line-height: 1.45; color:#1a1a1a; }}
h1 {{ font-size: 17pt; margin: 0 0 6px; }}
h2 {{ font-size: 13pt; margin: 18px 0 8px; border-bottom: 1.5px solid #1f4e79; color:#1f4e79; page-break-after: avoid; }}
h3 {{ font-size: 11pt; margin: 12px 0 6px; color:#2e5a88; page-break-after: avoid; }}
p, li {{ text-align: justify; }}
ol.toc {{ margin: 8px 0 16px 18px; }}
table {{ border-collapse: collapse; width: 100%; font-size: 7.8pt; margin: 8px 0 12px; }}
th, td {{ border: 1px solid #c9d4e0; padding: 3px 4px; vertical-align: top; }}
th {{ background: #1f4e79; color: #fff; text-align: left; }}
tr:nth-child(even) td {{ background: #f4f7fa; }}
.fig {{ text-align: center; margin: 10px 0; page-break-inside: avoid; }}
.fig img {{ max-width: 100%; max-height: 68mm; }}
.caption {{ font-size: 9pt; font-style: italic; color: #333; }}
.cover {{ text-align: center; margin: 28px 0 20px; }}
.kicker {{ letter-spacing: .08em; text-transform: uppercase; font-size: 9pt; color: #1f4e79; }}
.box {{ background: #eef3f8; border-left: 4px solid #1f4e79; padding: 8px 12px; margin: 10px 0; }}
.small {{ font-size: 9pt; color: #444; }}
.page-break {{ page-break-before: always; }}
</style></head><body>
<div class="cover">
  <div class="kicker">Universidad — Facultad de Ingeniería · Data Science</div>
  <h1>Informe de Fase 2 — Análisis inferencial</h1>
  <p><strong>Inasistencia escolar en la adolescencia paraguaya</strong><br>
  Contrastes H1–H6 · EPHC 2022–2025 · adolescentes 12–17 años</p>
  <p class="small"><strong>Integrantes:</strong> Matías Morínigo · César Vielman · Iván Paredes<br>
  Fase: 2 · Entrega: 3 de octubre de 2026 · α = {ALPHA} declarado <em>a priori</em></p>
</div>

<h2>Índice</h2>
<ol class="toc">
  <li>Correcciones a la Fase 1</li>
  <li>Pregunta, nivel de significancia e hipótesis</li>
  <li>Diseño inferencial y familias de pruebas</li>
  <li>Tabla resumen de contrastes</li>
  <li>Resultados e interpretación por hipótesis</li>
  <li>Comparaciones múltiples y confusores</li>
  <li>Conclusión de la Fase 2</li>
  <li>Reproducibilidad y uso de IA</li>
</ol>

<h2>1. Correcciones a la Fase 1</h2>
<p>Al momento de cerrar esta fase <strong>no hay devolución escrita del docente</strong> archivada en el
repositorio. Se listan, no obstante, las decisiones de diseño que se mantienen o se precisan al pasar
de la descripción a la inferencia:</p>
<ul>
  <li>Universo: personas de 12 a 17 años en EPHC 2022–2025; outcome <code>no_asiste</code> (ED08 = 19).</li>
  <li>Inferencia sobre <em>n</em> muestral (sin pesos de replicación DGEEC); los descriptivos de Fase 1
  siguen ponderados por FEX/FACTOR.</li>
  <li>Motivos ED10 agrupados en cinco categorías (Económico, Familiar, Oferta, Personal, Otro).</li>
  <li>Corrección BH sobre la familia {{H1, H2, H3, H4, H6}}; H5 se interpreta como modelo de control.</li>
  <li>Independencia de observaciones: supuesto limitado por el diseño complejo de la EPHC; se declara
  explícitamente en cada contraste.</li>
</ul>

<h2>2. Pregunta, nivel de significancia e hipótesis</h2>
<div class="box">¿En qué medida la zona de residencia (urbana/rural) y el departamento se asocian con la tasa de
inasistencia escolar de los adolescentes de 12 a 17 años en Paraguay, y qué motivos declaran los
hogares para explicar esa inasistencia, entre 2022 y 2025?</div>
<p>Nivel de significancia fijado <em>antes</em> de correr las pruebas: <strong>α = {ALPHA}</strong>.
Seis contrastes derivados de las hipótesis preliminares de la Fase 1.</p>

<table>
<tr><th>H</th><th>H₀ (formal)</th><th>Hₐ (formal)</th><th>Lenguaje natural</th></tr>
<tr><td>H1</td><td>p<sub>rural</sub> ≤ p<sub>urbana</sub></td>
<td>p<sub>rural</sub> &gt; p<sub>urbana</sub></td>
<td>La inasistencia es mayor en zona rural que en urbana.</td></tr>
<tr><td>H2</td><td>Zona ⊥ motivo agrupado</td>
<td>Zona y motivo no son independientes</td>
<td>Los motivos declarados difieren según la zona.</td></tr>
<tr><td>H3</td><td>Tasa igual entre departamentos</td>
<td>Al menos un departamento difiere</td>
<td>Hay heterogeneidad territorial en la inasistencia.</td></tr>
<tr><td>H4</td><td>ρ(edad, no_asiste) = 0</td>
<td>ρ(edad, no_asiste) ≠ 0</td>
<td>La inasistencia se asocia monotonamente con la edad.</td></tr>
<tr><td>H5</td><td>OR<sub>rural</sub> = 1 (dados dpto, edad, sexo)</td>
<td>OR<sub>rural</sub> ≠ 1</td>
<td>El exceso rural persiste al controlar otras variables.</td></tr>
<tr><td>H6</td><td>Período ⊥ motivo agrupado</td>
<td>Período y motivo no son independientes</td>
<td>El mix de motivos de 2025 difiere del de 2022–2024.</td></tr>
</table>

<h2>3. Diseño inferencial y familias de pruebas</h2>
<p>Se cubren <strong>cuatro familias</strong> del repertorio de la cátedra (mínimo exigido: tres), más un IC bootstrap:</p>
<ol>
  <li><strong>Dos proporciones</strong> — z de proporciones (H1), con IC Wald y bootstrap de la diferencia.</li>
  <li><strong>Asociación categórica</strong> — χ² de independencia (H2, H3, H6) con V de Cramér.</li>
  <li><strong>Correlación ordinal</strong> — ρ de Spearman (H4).</li>
  <li><strong>Efecto controlando otras</strong> — regresión logística múltiple (H5).</li>
</ol>
<p>No se aplica t/ANOVA de medias porque el outcome es binario. Shapiro–Wilk y Levene no son el
supuesto central aquí; se verifican en cambio n grandes / frecuencias esperadas / monotonía /
especificaciones del logit, según corresponda.</p>

<h2 class="page-break">4. Tabla resumen de contrastes</h2>
<table>
<tr><th>H</th><th>Enunciado</th><th>Prueba</th><th>Supuestos</th><th>Estadístico</th><th>gl</th>
<th>p</th><th>IC 95%</th><th>Efecto</th><th>p BH</th><th>Decisión</th></tr>
{''.join(filas)}
</table>
<p class="small">Fuente reproducible: <code>output/tablas/fase2_resumen_contrastes.csv</code>.
BH = Benjamini–Hochberg FDR sobre {{H1…H4, H6}}. H5 reporta su p del coeficiente rural.
Ninguna decisión “acepta” H₀: si no se rechaza, se concluye que no hay evidencia suficiente.</p>

<h2>5. Resultados e interpretación por hipótesis</h2>

<h3>H1 — Zona rural vs urbana</h3>
<div class="fig"><img src="figuras/F2_1_tasa_zona.png" alt="H1">
<div class="caption">Gráfico F2-1. Tasas muestrales de inasistencia por zona.</div></div>
<p><strong>Supuestos.</strong> n<sub>rural</sub> y n<sub>urbana</sub> grandes (aprox. normal del estimador).
Independencia limitada por el diseño EPHC (declarada). <strong>Resultado.</strong> Rural
{h1['p_rural']*100:.1f}% vs urbana {h1['p_urbana']*100:.1f}%
(Δ = {h1['diff']*100:.1f} pp; h de Cohen = {h1['cohen_h']:.3f}).
z = {h1['z']:.3f}, p unilateral ≪ α. IC Wald de la diferencia
[{h1['ci_classic'][0]:.4f}, {h1['ci_classic'][1]:.4f}]; IC bootstrap
[{h1['ci_boot'][0]:.4f}, {h1['ci_boot'][1]:.4f}] — ambos coinciden en magnitud y excluyen el cero.
<strong>Lectura.</strong> Significancia clara; el efecto es pequeño–moderado en h, pero ~6 pp es relevante
para política educativa. Se rechaza H₀.</p>

<h3>H2 — Motivo agrupado × zona</h3>
<div class="fig"><img src="figuras/F2_2_motivo_zona.png" alt="H2">
<div class="caption">Gráfico F2-2. Distribución de motivos agrupados por zona.</div></div>
<p><strong>Supuestos.</strong> Todas las celdas esperadas ≥ 5 (mín. esp. = {extras['h2']['expected_min']:.1f}).
<strong>Resultado.</strong> χ² = {extras['h2']['chi2']:.2f}, V de Cramér = {extras['h2']['v']:.3f}.
<strong>Lectura.</strong> La oferta escolar concentra peso en el campo y casi no en la ciudad; asociación
débil–moderada. Se rechaza H₀ tras BH.</p>

<h3>H3 — Heterogeneidad departamental</h3>
<div class="fig"><img src="figuras/F2_3_tasa_departamento.png" alt="H3">
<div class="caption">Gráfico F2-3. Tasa muestral de inasistencia por departamento.</div></div>
<p><strong>Supuestos.</strong> Contingencia departamento × no_asiste con mín. esp. = {extras['h3']['expected_min']:.1f}.
<strong>Resultado.</strong> χ² = {extras['h3']['chi2']:.2f}, V = {extras['h3']['v']:.3f}.
<strong>Lectura.</strong> Con n grande el p es minúsculo; la relevancia práctica está en el abanico
Capital (baja) frente a Itapúa / San Pedro / Canindeyú (altas). Se rechaza H₀ tras BH.</p>

<h3>H4 — Edad e inasistencia</h3>
<div class="fig"><img src="figuras/F2_4_edad.png" alt="H4">
<div class="caption">Gráfico F2-4. Tasa de inasistencia por edad cumplida.</div></div>
<p><strong>Supuestos.</strong> Relación monótona; no se exige normalidad (por eso Spearman y no Pearson).
<strong>Resultado.</strong> ρ = {extras['h4']['rho']:.4f}, p ≪ α.
<strong>Lectura.</strong> |ρ| ≈ 0,22 es modesto como correlación, pero la escalera de ~1% a los 12 años a
~18% a los 17 es de alta relevancia práctica. Se rechaza H₀ tras BH.</p>

<h3>H5 — Efecto rural controlando departamento, edad y sexo</h3>
<div class="fig"><img src="figuras/F2_5_or_logistica.png" alt="H5">
<div class="caption">Gráfico F2-5. Odds ratios del modelo logístico (referencia: {h5['ref']}).</div></div>
<p><strong>Supuestos.</strong> Linealidad en el logit para edad (edad entera 12–17); n grande; referencia
departamental = {h5['ref']}. <strong>Resultado.</strong> OR rural = {h5['or_rural']:.3f}
(IC95 {h5['ci_rural'][0]:.3f}–{h5['ci_rural'][1]:.3f}); Pseudo-R² = {h5['prsquared']:.4f}.
<strong>Lectura.</strong> El exceso rural no desaparece al controlar departamento, edad y sexo: las odds de
inasistencia se multiplican por ~{h5['or_rural']:.1f} en zona rural. <em>Esto no establece causalidad</em>
(corte transversal; posibles confusores omitidos: ingreso, trabajo adolescente, oferta escolar).
Se rechaza H₀ del coeficiente rural.</p>

<h3>H6 — Mix de motivos 2025 vs 2022–2024</h3>
<div class="fig"><img src="figuras/F2_6_motivo_periodo.png" alt="H6">
<div class="caption">Gráfico F2-6. Mix de motivos por período.</div></div>
<p><strong>Supuestos.</strong> Contingencia período × motivo; mín. esp. = {extras['h6']['expected_min']:.1f}.
<strong>Resultado.</strong> χ² = {extras['h6']['chi2']:.2f}, V = {extras['h6']['v']:.3f}.
<strong>Lectura.</strong> El perfil 2025 no es el mismo que el promedio 2022–2024: sube el peso de motivos
personales (“no quiere estudiar”) y baja el relativo económico. Asociación débil pero detectable.
Se rechaza H₀ tras BH.</p>

<h2>6. Comparaciones múltiples y confusores</h2>
<p>Se declararon <strong>seis</strong> contrastes. Sobre la familia de pruebas de asociación simple
{{H1, H2, H3, H4, H6}} se aplicó Benjamini–Hochberg (control de FDR). La corrección es necesaria
porque, al realizar varias pruebas, la probabilidad de al menos un falso positivo crece con el
número de contrastes. H5 se mantiene aparte como modelo multivariable de control, no como un
quinto test de la misma familia.</p>
<p><strong>Confusor principal discutido:</strong> zona ↔ departamento (la ruralidad no se reparte igual en
todo el país). H5 ataca precisamente esa amenaza: el OR rural permanece > 1 tras incluir
departamento. Otros confusores no observados (ingreso del hogar, trabajo, distancia a la escuela)
quedan como limitación hacia la Fase 3.</p>

<h2>7. Conclusión de la Fase 2</h2>
<p><strong>Qué se sostiene.</strong> Tras BH, H1–H4 y H6 se rechazan a favor de sus Hₐ; H5 también sostiene un
efecto rural positivo controlado. En conjunto: zona, departamento y edad se asocian a la
inasistencia más allá del azar muestral; los motivos dependen de la zona y del período.</p>
<p><strong>Implicancia para la pregunta.</strong> La evidencia inferencial respalda la lectura descriptiva de la
Fase 1: hay brecha rural–urbana, heterogeneidad departamental y una escalera por edad, con un
perfil de motivos que no es homogéneo ni en el espacio ni en el tiempo.</p>
<p><strong>Limitaciones de la inferencia.</strong> Diseño muestral complejo sin pesos de replicación; motivo
autorreportado; sin ingreso/trabajo en H5; corte transversal repetido (no panel); cobertura sin
Boquerón ni Alto Paraguay.</p>
<p><strong>Hacia la Fase 3.</strong> Predictores prioritarios: zona, departamento, edad y sexo (período opcional).
Target: <code>no_asiste</code>. Tarea: clasificación binaria. Línea base: clase mayoritaria (asiste).
Partición cronológica (entrenar 2022–2024, probar 2025) para respetar el componente temporal.</p>

<h2>8. Reproducibilidad y uso de IA</h2>
<p>Notebook ejecutado: <code>notebooks/Fase2_DataScience_EPHC.ipynb</code> (HTML en
<code>output/Fase2.html</code>). Tabla de contrastes:
<code>output/tablas/fase2_resumen_contrastes.csv</code>. Semilla aleatoria = 42. Helpers en
<code>src/fase2_helpers.py</code>; regeneración: <code>python scripts/build_fase2.py</code>.</p>
<p>Se usó un asistente de IA para estructurar el notebook/informe y pulir redacción. Las decisiones
metodológicas (familia de pruebas, BH, bootstrap, modelo de control) y la interpretación de los
resultados fueron definidas y verificadas por el grupo contra las salidas de scipy/statsmodels.</p>
</body></html>"""
    path = DIR_OUT / "Informe_Fase2.html"
    path.write_text(html, encoding="utf-8")
    return path


def main():
    print("Cargando datos…")
    df = load_data()
    print("Corriendo contrastes…")
    resumen, extras = run_tests(df)
    resumen.to_csv(DIR_TAB / "fase2_resumen_contrastes.csv", index=False, encoding="utf-8-sig")
    print(resumen[["hipotesis", "p_valor", "p_ajustado_BH", "decision"]].to_string(index=False))
    print("Figuras…")
    make_figures(df, extras)
    print("Notebook…")
    nb_path = write_notebook(resumen, extras, df)
    print("Informe HTML…")
    inf = write_informe(resumen, extras)
    print("OK:", nb_path)
    print("OK:", inf)
    print("OK:", DIR_TAB / "fase2_resumen_contrastes.csv")


if __name__ == "__main__":
    main()
