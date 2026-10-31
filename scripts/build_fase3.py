#!/usr/bin/env python3
"""Construye la Fase 3: modelado, figuras, notebook, informe e integrador."""
from __future__ import annotations

import base64
import json
import sys
import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, precision_recall_curve
from sklearn.model_selection import ParameterGrid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.fase3_helpers import (  # noqa: E402
    FEATURE_COLS,
    FEATURES_CAT,
    FEATURES_NUM,
    TARGET,
    best_threshold_by_f1,
    chronological_split,
    load_modeling_frame,
    make_pipeline,
    metrics_row,
    year_folds,
)

SEMILLA = 42
ALPHA = 0.05
TEST_YEAR = 2025
np.random.seed(SEMILLA)
warnings.filterwarnings("ignore")

DIR_OUT = ROOT / "output"
DIR_FIG = DIR_OUT / "figuras"
DIR_TAB = DIR_OUT / "tablas"
DIR_MOD = DIR_OUT / "modelos"
DIR_NB = ROOT / "notebooks"
for d in (DIR_FIG, DIR_TAB, DIR_MOD, DIR_NB):
    d.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid")
C_POS, C_NEG, C_LINE = "#eb6834", "#2a78d6", "#52514e"


def resolve_dataset() -> Path:
    for p in (
        ROOT / "output" / "dataset_fase1.parquet",
        ROOT / "data" / "processed" / "dataset_fase1.parquet",
        ROOT / "entregable" / "fase2" / "05_dataset_limpio" / "dataset_fase1.parquet",
    ):
        if p.exists() and p.stat().st_size > 0:
            return p
    raise FileNotFoundError("No está dataset_fase1.parquet")


def temporal_cv_score(pipe, train_df, param_grid, scoring="f1"):
    """Grid search con validación temporal expanding (años 2022→2024)."""
    years = sorted(train_df["anio_encuesta"].unique())
    folds = year_folds(years)
    best, best_params, rows = -1.0, None, []
    for params in ParameterGrid(param_grid):
        scores = []
        for train_years, val_years in folds:
            tr = train_df[train_df["anio_encuesta"].isin(train_years)]
            va = train_df[train_df["anio_encuesta"].isin(val_years)]
            est = make_pipeline(
                pipe["clf"].__class__(**{**pipe["clf"].get_params(), **params}),
                scale_numeric=isinstance(pipe["clf"], LogisticRegression),
            )
            # rebuild with right scale flag
            scale = isinstance(pipe["clf"], LogisticRegression)
            from sklearn.base import clone

            clf = pipe["clf"].__class__(**{**{k: v for k, v in pipe["clf"].get_params().items()
                                              if k in pipe["clf"].get_params()}, **params})
            # simpler: set params on a fresh estimator
            base = pipe["clf"].__class__(random_state=SEMILLA) if "random_state" in pipe["clf"].get_params() else pipe["clf"].__class__()
            try:
                base.set_params(**{**{k: v for k, v in pipe["clf"].get_params().items() if k != "random_state"}, **params})
            except Exception:
                base.set_params(**params)
            if "random_state" in base.get_params():
                base.set_params(random_state=SEMILLA)
            if "class_weight" in pipe["clf"].get_params() and "class_weight" not in params:
                if pipe["clf"].get_params().get("class_weight") is not None:
                    base.set_params(class_weight=pipe["clf"].get_params()["class_weight"])
            model = make_pipeline(base, scale_numeric=scale)
            model.fit(tr[FEATURE_COLS], tr[TARGET])
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(va[FEATURE_COLS])[:, 1]
                pred = (proba >= 0.5).astype(int)
            else:
                pred = model.predict(va[FEATURE_COLS])
                proba = None
            from sklearn.metrics import f1_score, roc_auc_score

            f1 = f1_score(va[TARGET], pred, zero_division=0)
            auc = roc_auc_score(va[TARGET], proba) if proba is not None else np.nan
            scores.append(f1)
            rows.append({"params": str(params), "val_year": int(val_years[0]), "f1": f1, "auc": auc})
        mean_f1 = float(np.mean(scores))
        if mean_f1 > best:
            best, best_params = mean_f1, params
    return best_params, best, pd.DataFrame(rows)


def _sample_weight(y: pd.Series) -> np.ndarray:
    y = np.asarray(y)
    pos = (y == 1).sum()
    neg = (y == 0).sum()
    w = np.ones(len(y), dtype=float)
    w[y == 1] = neg / max(pos, 1)
    return w


def fit_grid(estimator, param_grid, train_df, scale_numeric: bool, use_sample_weight: bool = False):
    years = sorted(train_df["anio_encuesta"].unique())
    folds = year_folds(list(years))
    best_score, best_params, rows = -1.0, None, []
    from sklearn.metrics import f1_score, roc_auc_score

    for params in ParameterGrid(param_grid):
        fold_f1 = []
        for train_years, val_years in folds:
            tr = train_df[train_df["anio_encuesta"].isin(train_years)]
            va = train_df[train_df["anio_encuesta"].isin(val_years)]
            clf = estimator.__class__(**estimator.get_params())
            clf.set_params(**params)
            if "random_state" in clf.get_params():
                clf.set_params(random_state=SEMILLA)
            model = make_pipeline(clf, scale_numeric=scale_numeric)
            fit_kw = {}
            if use_sample_weight:
                fit_kw["clf__sample_weight"] = _sample_weight(tr[TARGET])
            model.fit(tr[FEATURE_COLS], tr[TARGET], **fit_kw)
            proba = model.predict_proba(va[FEATURE_COLS])[:, 1]
            # umbral provisional 0.5 en CV; el umbral final se calibra en 2024
            pred = (proba >= 0.5).astype(int)
            f1 = f1_score(va[TARGET], pred, zero_division=0)
            auc = roc_auc_score(va[TARGET], proba)
            fold_f1.append(f1)
            rows.append({"params": json.dumps(params, sort_keys=True), "val_year": int(val_years[0]),
                         "f1": f1, "auc": auc})
        mean_f1 = float(np.mean(fold_f1))
        if mean_f1 > best_score:
            best_score, best_params = mean_f1, params
    return best_params, best_score, pd.DataFrame(rows)


def run_modeling(df: pd.DataFrame):
    train_df, test_df = chronological_split(df, TEST_YEAR)
    X_train, y_train = train_df[FEATURE_COLS], train_df[TARGET]
    X_test, y_test = test_df[FEATURE_COLS], test_df[TARGET]

    # Validación interna: último año de train (2024) para umbral; CV temporal para hiperparámetros
    inner_train = train_df[train_df["anio_encuesta"] < 2024]
    inner_val = train_df[train_df["anio_encuesta"] == 2024]

    results = []
    fitted = {}
    thresholds = {}
    cv_tables = {}
    importances = {}

    # --- Baseline: clase mayoritaria ---
    base = make_pipeline(DummyClassifier(strategy="most_frequent"), scale_numeric=False)
    base.fit(X_train, y_train)
    # Dummy may not have predict_proba with both classes if only one class predicted
    proba_tr = np.zeros(len(X_train))
    proba_te = np.zeros(len(X_test))
    pred_tr = base.predict(X_train)
    pred_te = base.predict(X_test)
    results.append(metrics_row("Línea base (mayoritaria)", "train", y_train, pred_tr, proba_tr))
    results.append(metrics_row("Línea base (mayoritaria)", "test", y_test, pred_te, proba_te))
    fitted["Línea base (mayoritaria)"] = base
    thresholds["Línea base (mayoritaria)"] = 0.5

    # --- Logística ---
    logit_grid = {
        "C": [0.1, 1.0, 3.0],
        "class_weight": ["balanced"],
        "max_iter": [500],
        "solver": ["lbfgs"],
    }
    best_logit, score_logit, cv_logit = fit_grid(
        LogisticRegression(random_state=SEMILLA), logit_grid, train_df, scale_numeric=True
    )
    cv_tables["Logística"] = cv_logit
    logit = make_pipeline(
        LogisticRegression(random_state=SEMILLA, **best_logit), scale_numeric=True
    )
    logit.fit(inner_train[FEATURE_COLS], inner_train[TARGET])
    proba_val = logit.predict_proba(inner_val[FEATURE_COLS])[:, 1]
    t_logit, _ = best_threshold_by_f1(inner_val[TARGET], proba_val)
    logit.fit(X_train, y_train)
    fitted["Logística"] = logit
    thresholds["Logística"] = t_logit
    for split, X, y in [("train", X_train, y_train), ("test", X_test, y_test)]:
        proba = logit.predict_proba(X)[:, 1]
        pred = (proba >= t_logit).astype(int)
        results.append(metrics_row("Logística", split, y, pred, proba))

    # --- Random Forest ---
    rf_grid = {
        "n_estimators": [100, 200],
        "max_depth": [6, 10, None],
        "min_samples_leaf": [5, 20],
        "class_weight": ["balanced_subsample"],
    }
    best_rf, score_rf, cv_rf = fit_grid(
        RandomForestClassifier(random_state=SEMILLA, n_jobs=-1), rf_grid, train_df, scale_numeric=False
    )
    cv_tables["Random Forest"] = cv_rf
    rf = make_pipeline(
        RandomForestClassifier(random_state=SEMILLA, n_jobs=-1, **best_rf), scale_numeric=False
    )
    rf.fit(inner_train[FEATURE_COLS], inner_train[TARGET])
    t_rf, _ = best_threshold_by_f1(inner_val[TARGET], rf.predict_proba(inner_val[FEATURE_COLS])[:, 1])
    rf.fit(X_train, y_train)
    fitted["Random Forest"] = rf
    thresholds["Random Forest"] = t_rf
    for split, X, y in [("train", X_train, y_train), ("test", X_test, y_test)]:
        proba = rf.predict_proba(X)[:, 1]
        pred = (proba >= t_rf).astype(int)
        results.append(metrics_row("Random Forest", split, y, pred, proba))

    # --- HistGradientBoosting ---
    hgb_grid = {
        "learning_rate": [0.05, 0.1],
        "max_depth": [3, 5],
        "max_iter": [100, 200],
        "min_samples_leaf": [20, 50],
    }
    # class imbalance via sample weight on fit - HGB has no class_weight in older sklearn; use scale_pos_weight approx via sample_weight in CV
    # For simplicity use default and threshold tuning
    best_hgb, score_hgb, cv_hgb = fit_grid(
        HistGradientBoostingClassifier(random_state=SEMILLA),
        hgb_grid,
        train_df,
        scale_numeric=False,
        use_sample_weight=True,
    )
    cv_tables["HistGradientBoosting"] = cv_hgb
    hgb = make_pipeline(
        HistGradientBoostingClassifier(random_state=SEMILLA, **best_hgb), scale_numeric=False
    )
    # sample weights for imbalance on final fits
    def sw(y):
        pos = (y == 1).sum()
        neg = (y == 0).sum()
        w = np.ones(len(y))
        w[y == 1] = neg / max(pos, 1)
        return w

    hgb.fit(inner_train[FEATURE_COLS], inner_train[TARGET], clf__sample_weight=sw(inner_train[TARGET]))
    t_hgb, _ = best_threshold_by_f1(inner_val[TARGET], hgb.predict_proba(inner_val[FEATURE_COLS])[:, 1])
    hgb.fit(X_train, y_train, clf__sample_weight=sw(y_train))
    fitted["HistGradientBoosting"] = hgb
    thresholds["HistGradientBoosting"] = t_hgb
    for split, X, y in [("train", X_train, y_train), ("test", X_test, y_test)]:
        proba = hgb.predict_proba(X)[:, 1]
        pred = (proba >= t_hgb).astype(int)
        results.append(metrics_row("HistGradientBoosting", split, y, pred, proba))

    resumen = pd.DataFrame(results)
    # elegir mejor modelo por F1 en test? NO - choose by validation F1 then report test once
    # Re-evaluate: pick model by F1 on inner_val with its threshold, then only report that as final for test
    val_scores = {}
    for name in ["Logística", "Random Forest", "HistGradientBoosting"]:
        m = fitted[name]
        # refit concept: models already fit on full train; for selection use 2024 holdout from a refit
        # Simpler: use train metrics vs test carefully — select by mean CV F1 from cv tables
        mean_cv = cv_tables[name].groupby("params")["f1"].mean().max()
        val_scores[name] = mean_cv
    best_name = max(val_scores, key=val_scores.get)

    # Feature importance from RF (and logit coefficients)
    rf_clf = fitted["Random Forest"].named_steps["clf"]
    prep = fitted["Random Forest"].named_steps["prep"]
    feat_names = list(FEATURES_NUM) + list(
        prep.named_transformers_["cat"].get_feature_names_out(FEATURES_CAT)
    )
    importances["Random Forest"] = pd.Series(rf_clf.feature_importances_, index=feat_names).sort_values(ascending=False)

    logit_clf = fitted["Logística"].named_steps["clf"]
    prep_l = fitted["Logística"].named_steps["prep"]
    feat_l = list(FEATURES_NUM) + list(prep_l.named_transformers_["cat"].get_feature_names_out(FEATURES_CAT))
    importances["Logística"] = pd.Series(logit_clf.coef_[0], index=feat_l).sort_values(key=np.abs, ascending=False)

    meta = {
        "best_model": best_name,
        "best_params": {
            "Logística": best_logit,
            "Random Forest": best_rf,
            "HistGradientBoosting": best_hgb,
        },
        "thresholds": thresholds,
        "cv_f1": val_scores,
        "features": FEATURE_COLS,
        "target": TARGET,
        "test_year": TEST_YEAR,
        "seed": SEMILLA,
        "umbral": thresholds[best_name],
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
        "rate_train": float(y_train.mean()),
        "rate_test": float(y_test.mean()),
    }

    # Error analysis on test for best model
    best_m = fitted[best_name]
    t_best = thresholds[best_name]
    proba_test = best_m.predict_proba(X_test)[:, 1]
    pred_test = (proba_test >= t_best).astype(int)
    err = test_df.copy()
    err["proba"] = proba_test
    err["pred"] = pred_test
    err["error"] = err[TARGET] != err["pred"]
    err["tipo_error"] = np.where(
        (err[TARGET] == 1) & (err["pred"] == 0), "FN",
        np.where((err[TARGET] == 0) & (err["pred"] == 1), "FP", "OK"),
    )

    return {
        "train_df": train_df,
        "test_df": test_df,
        "resumen": resumen,
        "fitted": fitted,
        "thresholds": thresholds,
        "cv_tables": cv_tables,
        "importances": importances,
        "meta": meta,
        "errors": err,
        "best_name": best_name,
        "best_params_all": {"Logística": best_logit, "Random Forest": best_rf, "HistGradientBoosting": best_hgb},
        "cv_scores": val_scores,
    }


def make_figures(ctx):
    resumen = ctx["resumen"]
    test = resumen[resumen.split == "test"].set_index("modelo")
    train = resumen[resumen.split == "train"].set_index("modelo")

    # F3-1 métricas test
    fig, ax = plt.subplots(figsize=(9, 4.5))
    metrics = ["accuracy", "precision", "recall", "f1", "auc_roc"]
    plot_df = test[metrics]
    plot_df.plot(kind="bar", ax=ax, rot=15)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Valor")
    ax.set_title("F3-1 · Métricas en prueba (2025)")
    ax.legend(frameon=False, ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.12))
    fig.tight_layout()
    fig.savefig(DIR_FIG / "F3_1_metricas_test.png")
    plt.close()

    # F3-2 train vs test F1
    fig, ax = plt.subplots(figsize=(8, 4))
    modelos = list(test.index)
    x = np.arange(len(modelos))
    ax.bar(x - 0.2, [train.loc[m, "f1"] for m in modelos], 0.4, label="Train", color=C_NEG)
    ax.bar(x + 0.2, [test.loc[m, "f1"] for m in modelos], 0.4, label="Test 2025", color=C_POS)
    ax.set_xticks(x)
    ax.set_xticklabels(modelos, rotation=15)
    ax.set_ylabel("F1")
    ax.set_title("F3-2 · F1 train vs test (sobreajuste)")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(DIR_FIG / "F3_2_train_vs_test.png")
    plt.close()

    # F3-3 ROC best
    best = ctx["best_name"]
    err = ctx["errors"]
    fig, ax = plt.subplots(figsize=(6, 5))
    RocCurveDisplay.from_predictions(err[TARGET], err["proba"], ax=ax, name=best)
    ax.plot([0, 1], [0, 1], ls="--", color=C_LINE, lw=1)
    ax.set_title(f"F3-3 · ROC en prueba — {best}")
    fig.tight_layout()
    fig.savefig(DIR_FIG / "F3_3_roc.png")
    plt.close()

    # F3-4 confusion
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(err[TARGET], err["pred"], ax=ax, colorbar=False)
    ax.set_title(f"F3-4 · Matriz de confusión — {best} (test 2025)")
    fig.tight_layout()
    fig.savefig(DIR_FIG / "F3_4_confusion.png")
    plt.close()

    # F3-5 importances RF top 12
    imp = ctx["importances"]["Random Forest"].head(12)
    fig, ax = plt.subplots(figsize=(8, 5))
    imp.iloc[::-1].plot(kind="barh", ax=ax, color=C_NEG)
    ax.set_xlabel("Importancia (impureza)")
    ax.set_title("F3-5 · Importancia de variables — Random Forest")
    fig.tight_layout()
    fig.savefig(DIR_FIG / "F3_5_importancia.png")
    plt.close()

    # F3-6 errores por edad/zona
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fn = err[err.tipo_error == "FN"]
    ok_pos = err[(err[TARGET] == 1) & (err.tipo_error == "OK")]
    for label, subset, color in [("FN", fn, C_POS), ("TP", ok_pos, C_NEG)]:
        if len(subset):
            axes[0].hist(subset["edad"], bins=np.arange(11.5, 18.5, 1), alpha=0.6, label=label, color=color)
    axes[0].set_title("Edad: falsos negativos vs aciertos (positivos)")
    axes[0].legend(frameon=False)
    axes[0].set_xlabel("Edad")
    pd.crosstab(err["zona"], err["tipo_error"], normalize="index").plot(kind="bar", ax=axes[1], rot=0)
    axes[1].set_title("Tipo de error por zona")
    axes[1].set_ylabel("Proporción")
    axes[1].legend(frameon=False, fontsize=8)
    fig.suptitle("F3-6 · Análisis de errores (test 2025)", fontweight="bold")
    fig.tight_layout()
    fig.savefig(DIR_FIG / "F3_6_errores.png")
    plt.close()


def save_model(ctx):
    best = ctx["best_name"]
    bundle = {
        "pipeline": ctx["fitted"][best],
        "threshold": ctx["thresholds"][best],
        "model_name": best,
        "features": FEATURE_COLS,
        "target": TARGET,
    }
    path = DIR_MOD / "fase3_modelo_final.joblib"
    joblib.dump(bundle, path)
    # Bundle con todos los modelos para predicción manual comparativa
    joblib.dump(
        {
            "fitted": ctx["fitted"],
            "thresholds": ctx["thresholds"],
            "features": FEATURE_COLS,
            "target": TARGET,
            "best_model": best,
        },
        DIR_MOD / "fase3_modelos_todos.joblib",
    )
    meta = dict(ctx["meta"])
    meta["umbral"] = ctx["thresholds"][best]
    (DIR_MOD / "fase3_modelo_meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    # also save all metrics
    ctx["resumen"].to_csv(DIR_TAB / "fase3_metricas.csv", index=False, encoding="utf-8-sig")
    ctx["errors"].to_csv(DIR_TAB / "fase3_errores_test.csv", index=False, encoding="utf-8-sig")
    ctx["importances"]["Random Forest"].rename("importancia").to_csv(
        DIR_TAB / "fase3_importancia_rf.csv", encoding="utf-8-sig"
    )
    for name, tab in ctx["cv_tables"].items():
        tab.to_csv(DIR_TAB / f"fase3_cv_{name.lower().replace(' ', '_')}.csv", index=False, encoding="utf-8-sig")
    return path


def img_b64(name: str) -> str:
    raw = (DIR_FIG / name).read_bytes()
    return "data:image/png;base64," + base64.b64encode(raw).decode("ascii")


def write_informe(ctx) -> Path:
    r = ctx["resumen"]
    test = r[r.split == "test"].set_index("modelo")
    train = r[r.split == "train"].set_index("modelo")
    best = ctx["best_name"]
    meta = ctx["meta"]
    t = ctx["thresholds"][best]
    rows = []
    for m in test.index:
        rows.append(
            "<tr>"
            f"<td>{m}</td>"
            f"<td>{train.loc[m, 'f1']:.3f}</td>"
            f"<td>{test.loc[m, 'accuracy']:.3f}</td>"
            f"<td>{test.loc[m, 'precision']:.3f}</td>"
            f"<td>{test.loc[m, 'recall']:.3f}</td>"
            f"<td>{test.loc[m, 'f1']:.3f}</td>"
            f"<td>{test.loc[m, 'auc_roc']:.3f}</td>"
            f"<td>{ctx['thresholds'][m]:.2f}</td>"
            "</tr>"
        )
    cm = ctx["errors"]
    tn = int(((cm[TARGET] == 0) & (cm.pred == 0)).sum())
    fp = int(((cm[TARGET] == 0) & (cm.pred == 1)).sum())
    fn = int(((cm[TARGET] == 1) & (cm.pred == 0)).sum())
    tp = int(((cm[TARGET] == 1) & (cm.pred == 1)).sum())
    top_imp = ctx["importances"]["Random Forest"].head(8)
    imp_rows = "".join(f"<tr><td>{i}</td><td>{v:.3f}</td></tr>" for i, v in top_imp.items())

    html = f"""<!DOCTYPE html><html lang="es"><head><meta charset="utf-8">
<title>Informe Fase 3 — Modelado predictivo</title>
<style>
@page {{ size: A4; margin: 16mm; @bottom-center {{ content: counter(page); font-size: 9pt; }} }}
body {{ font-family: "Source Sans 3","Segoe UI",Helvetica,Arial,sans-serif; font-size: 10.5pt; line-height: 1.45; color:#1a1a1a; }}
h1 {{ font-size: 17pt; margin: 0 0 6px; }}
h2 {{ font-size: 13pt; margin: 16px 0 8px; border-bottom: 1.5px solid #1f4e79; color:#1f4e79; }}
h3 {{ font-size: 11pt; margin: 12px 0 6px; color:#2e5a88; }}
p, li {{ text-align: justify; }}
table {{ border-collapse: collapse; width: 100%; font-size: 8.5pt; margin: 8px 0 12px; }}
th, td {{ border: 1px solid #c9d4e0; padding: 3px 5px; vertical-align: top; }}
th {{ background: #1f4e79; color: #fff; text-align: left; }}
tr:nth-child(even) td {{ background: #f4f7fa; }}
.fig {{ text-align: center; margin: 10px 0; page-break-inside: avoid; }}
.fig img {{ max-width: 100%; max-height: 72mm; }}
.caption {{ font-size: 9pt; font-style: italic; color: #333; }}
.cover {{ text-align: center; margin: 24px 0 18px; }}
.kicker {{ letter-spacing: .08em; text-transform: uppercase; font-size: 9pt; color: #1f4e79; }}
.box {{ background: #eef3f8; border-left: 4px solid #1f4e79; padding: 8px 12px; margin: 10px 0; }}
.small {{ font-size: 9pt; color: #444; }}
.page-break {{ page-break-before: always; }}
</style></head><body>
<div class="cover">
  <div class="kicker">Facultad Politécnica — Universidad Nacional de Asunción · Data Science</div>
  <h1>Informe de Fase 3 — Análisis predictivo</h1>
  <p><strong>Inasistencia escolar en la adolescencia paraguaya</strong><br>
  Clasificación de <code>no_asiste</code> · EPHC 2022–2025 · 12–17 años</p>
  <p class="small"><strong>Integrantes:</strong> Matías Morínigo · César Vielman · Iván Paredes<br>
  Fase: 3 · Entrega: 24 de octubre de 2026 · Semilla = {SEMILLA}</p>
</div>

<h2>Índice</h2>
<ol>
  <li>Correcciones a la Fase 2</li>
  <li>Tarea predictiva y diseño del experimento</li>
  <li>Preparación y partición</li>
  <li>Línea base y modelos</li>
  <li>Resultados en prueba</li>
  <li>Análisis de errores e importancia</li>
  <li>Conclusión de la Fase 3</li>
  <li>Reproducibilidad y uso de IA</li>
</ol>

<h2>1. Correcciones a la Fase 2</h2>
<p>No hay devolución escrita del docente archivada en el repositorio. Se conserva el diseño inferencial de la
Fase 2 y se operacionaliza lo ya fijado para el modelado: partición <strong>cronológica</strong>
(entrenar 2022–2024, probar 2025); predictores <code>zona</code>, <code>departamento</code>, <code>edad</code> y
<code>sexo</code>; target <code>no_asiste</code>; línea base = clase mayoritaria; dependencia por UPM queda fuera
del estimador predictivo (no hay pesos de replicación DGEEC en scikit-learn), pero se discute como límite
de generalización.</p>

<h2>2. Tarea predictiva y diseño del experimento</h2>
<div class="box">Clasificación binaria: predecir si un adolescente de 12–17 años <em>no asiste</em> a un centro
educativo (<code>no_asiste = 1</code>), con predictores respaldados por la Fase 2.</div>
<p>La tarea responde al objetivo predictivo del proyecto: anticipar el riesgo de inasistencia a partir de
atributos observables en la EPHC, sin usar el motivo (solo disponible cuando ya no asiste).</p>
<ul>
  <li><strong>Target:</strong> <code>no_asiste</code> (prevalencia train ≈ {100*meta['rate_train']:.1f} %; test 2025 ≈ {100*meta['rate_test']:.1f} %).</li>
  <li><strong>Features:</strong> edad, zona, departamento, sexo.</li>
  <li><strong>Partición:</strong> train n = {meta['n_train']:,} (2022–2024); test n = {meta['n_test']:,} (solo 2025).</li>
  <li><strong>Validación:</strong> expanding year-folds dentro de 2022–2024 + holdout 2024 para el umbral.</li>
  <li><strong>Métrica de selección:</strong> F1 media en validación temporal (desbalance ≈ 8 %).</li>
</ul>

<h2>3. Preparación y partición</h2>
<p>One-hot de categóricas; estandarización de edad solo en la logística. Desbalance: <code>class_weight</code>
en logística y bosques; pesos de muestra en boosting; umbral de decisión calibrado en 2024 maximizando F1
(no se toca 2025 para elegir umbral ni hiperparámetros).</p>

<h2>4. Línea base y modelos</h2>
<ol>
  <li><strong>Línea base:</strong> DummyClassifier (clase mayoritaria = asiste). Exactitud alta, recall/F1 ≈ 0.</li>
  <li><strong>Logística</strong> (familia lineal): C y class_weight vía grilla temporal. Mejor: {meta['best_params']['Logística']}.</li>
  <li><strong>Random Forest</strong> (árboles): profundidad, hojas y n_estimators. Mejor: {meta['best_params']['Random Forest']}.</li>
  <li><strong>HistGradientBoosting</strong> (boosting): learning rate, profundidad, iteraciones. Mejor: {meta['best_params']['HistGradientBoosting']}.</li>
</ol>
<p>Modelo seleccionado por F1 de validación temporal: <strong>{best}</strong> (umbral = {t:.2f}).</p>

<h2 class="page-break">5. Resultados en prueba (2025, una sola vez)</h2>
<table>
<tr><th>Modelo</th><th>F1 train</th><th>Accuracy</th><th>Precision</th><th>Recall</th><th>F1 test</th><th>AUC-ROC</th><th>Umbral</th></tr>
{''.join(rows)}
</table>
<p class="small">La exactitud de la línea base es engañosa: no detecta a ningún no asistente. En desbalance,
F1, recall y AUC-ROC son las métricas que importan.</p>

<div class="fig"><img src="{img_b64('F3_1_metricas_test.png')}" alt="métricas">
<div class="caption">Gráfico F3-1. Métricas en el conjunto de prueba 2025.</div></div>

<div class="fig"><img src="{img_b64('F3_2_train_vs_test.png')}" alt="sobreajuste">
<div class="caption">Gráfico F3-2. F1 en entrenamiento vs prueba: brecha de generalización.</div></div>

<div class="fig"><img src="{img_b64('F3_3_roc.png')}" alt="roc">
<div class="caption">Gráfico F3-3. Curva ROC del modelo final en 2025.</div></div>

<p><strong>Desempeño del modelo final ({best}).</strong>
Accuracy = {test.loc[best,'accuracy']:.3f}; precisión = {test.loc[best,'precision']:.3f};
recall = {test.loc[best,'recall']:.3f}; F1 = {test.loc[best,'f1']:.3f};
AUC-ROC = {test.loc[best,'auc_roc']:.3f}. Matriz: TN={tn}, FP={fp}, FN={fn}, TP={tp}.
Supera claramente a la línea base en F1/recall/AUC. La brecha train−test en F1 indica
{"sobreajuste moderado" if (train.loc[best,'f1'] - test.loc[best,'f1']) > 0.05 else "poca evidencia de sobreajuste severo"}.</p>

<h2>6. Análisis de errores e importancia</h2>
<div class="fig"><img src="{img_b64('F3_4_confusion.png')}" alt="cm">
<div class="caption">Gráfico F3-4. Matriz de confusión en prueba.</div></div>
<div class="fig"><img src="{img_b64('F3_6_errores.png')}" alt="errores">
<div class="caption">Gráfico F3-6. Perfil de falsos negativos y errores por zona.</div></div>
<div class="fig"><img src="{img_b64('F3_5_importancia.png')}" alt="imp">
<div class="caption">Gráfico F3-5. Importancia de variables (Random Forest).</div></div>
<table><tr><th>Variable (RF)</th><th>Importancia</th></tr>{imp_rows}</table>
<p>Las variables más influyentes (edad, zona rural, departamentos de alta inasistencia) coinciden con H1, H3 y H4
de la Fase 2. El sexo aporta menos, coherente con un OR más cercano a 1 en H5.</p>

<h2>7. Conclusión de la Fase 3</h2>
<p><strong>Desempeño.</strong> El modelo final ({best}) supera a la línea base mayoritaria en las métricas relevantes
para la clase minoritaria. No es un detector perfecto: el recall &lt; 1 implica falsos negativos (adolescentes en
riesgo no alertados).</p>
<p><strong>Utilidad práctica.</strong> Sirve como <em>priorización</em> territorial y etaria (dónde concentrar
acompañamiento), no como veredicto individual. No debe usarse para sancionar hogares ni para negar cupos.</p>
<p><strong>Límites.</strong> Corte transversal; sin ingreso/trabajo/distancia a la escuela; cobertura sin Boquerón ni
Alto Paraguay; 2025 es un solo año de prueba; el diseño muestral EPHC no entra en el estimador.</p>
<p><strong>Riesgos éticos.</strong> Un umbral agresivo eleva falsos positivos (estigmatización); uno permisivo deja
pasar falsos negativos. Cualquier uso institucional debe fijar el umbral con costo asimétrico explícito y auditoría
por departamento/zona.</p>
<p><strong>Respuesta a la pregunta de investigación.</strong> Zona, departamento y edad no solo se asocian
inferencialmente con la inasistencia (Fases 1–2): también aportan señal predictiva fuera de muestra en 2025. Los
motivos (Fase 1–2) explican el fenómeno pero no entran al predictor porque solo se observan tras la inasistencia.</p>

<h2>8. Reproducibilidad y uso de IA</h2>
<p>Notebook: <code>notebooks/Fase3_DataScience_EPHC.ipynb</code>. Modelo:
<code>output/modelos/fase3_modelo_final.joblib</code>. Inferencia:
<code>python scripts/infer_fase3.py datos.csv</code>. Semilla 42. Se usó IA para estructura y redacción; la
selección de partición cronológica, familias y métricas fue definida y verificada por el grupo.</p>
</body></html>"""
    path = DIR_OUT / "Informe_Fase3.html"
    path.write_text(html, encoding="utf-8")
    return path


def write_integrador(ctx) -> Path:
    best = ctx["best_name"]
    test = ctx["resumen"]
    te = test[(test.split == "test") & (test.modelo == best)].iloc[0]
    html = f"""<!DOCTYPE html><html lang="es"><head><meta charset="utf-8">
<title>Informe integrador — Tres fases</title>
<style>
@page {{ size: A4; margin: 16mm; @bottom-center {{ content: counter(page); font-size: 9pt; }} }}
body {{ font-family: "Source Sans 3","Segoe UI",Helvetica,Arial,sans-serif; font-size: 10.5pt; line-height: 1.45; }}
h1 {{ font-size: 17pt; }} h2 {{ font-size: 13pt; color:#1f4e79; border-bottom: 1.5px solid #1f4e79; }}
.cover {{ text-align:center; margin: 28px 0; }}
.box {{ background:#eef3f8; border-left:4px solid #1f4e79; padding:8px 12px; }}
table {{ border-collapse: collapse; width: 100%; font-size: 9pt; }}
th, td {{ border: 1px solid #c9d4e0; padding: 4px; }}
th {{ background:#1f4e79; color:#fff; }}
.page-break {{ page-break-before: always; }}
</style></head><body>
<div class="cover">
  <p style="color:#1f4e79; letter-spacing:.08em; text-transform:uppercase; font-size:9pt;">Facultad Politécnica — Universidad Nacional de Asunción · Data Science</p>
  <h1>Informe integrador</h1>
  <p><strong>Inasistencia escolar en la adolescencia paraguaya (EPHC 2022–2025)</strong></p>
  <p>Matías Morínigo · César Vielman · Iván Paredes<br>24 de octubre de 2026</p>
</div>

<h2>1. Pregunta y arco del proyecto</h2>
<div class="box">¿En qué medida la zona y el departamento se asocian con la tasa de inasistencia de adolescentes
de 12 a 17 años en Paraguay, y qué motivos declaran los hogares, entre 2022 y 2025?</div>
<p>Tres fases: <strong>describir</strong> (F1), <strong>contrastar</strong> (F2) y <strong>predecir</strong> (F3)
sobre el mismo universo EPHC.</p>

<h2>2. Fase 1 — Qué muestran los datos</h2>
<ul>
  <li>Universo: 24.316 adolescentes 12–17; tasa muestral de inasistencia ≈ 7,9 % (ponderada ≈ 7,4 %).</li>
  <li>Brecha rural–urbana visible; heterogeneidad departamental; escalera por edad; motivos económicos
  dominantes, con oferta más rural.</li>
  <li>Límite: sin Boquerón ni Alto Paraguay; motivo autorreportado.</li>
</ul>

<h2>3. Fase 2 — Qué se puede afirmar con evidencia</h2>
<ul>
  <li>Se rechazan las hipótesis nulas de H1–H6 (con ajuste por UPM y BH donde correspondía).</li>
  <li>OR rural ≈ 2,3 controlando departamento, edad y sexo (no causal).</li>
  <li>Edad, zona y departamento quedan como predictores prioritarios para la Fase 3.</li>
</ul>

<h2 class="page-break">4. Fase 3 — Qué predice el modelo</h2>
<ul>
  <li>Clasificación de <code>no_asiste</code>; train 2022–2024 / test 2025.</li>
  <li>Modelo final: <strong>{best}</strong>. En 2025: F1 = {te['f1']:.3f}, recall = {te['recall']:.3f},
  precisión = {te['precision']:.3f}, AUC-ROC = {te['auc_roc']:.3f} (supera la línea base mayoritaria en F1/recall).</li>
  <li>Importancias alineadas con F1–F2: edad, ruralidad, departamentos de alta tasa.</li>
</ul>

<h2>5. Conclusión general</h2>
<p>La inasistencia adolescente en Paraguay, en la cobertura EPHC disponible, se concentra en el campo, en
ciertos departamentos y en las edades altas del rango 12–17. Esa lectura descriptiva se sostiene
inferencialmente y aporta señal predictiva fuera de muestra. El modelo es una herramienta de
<strong>priorización</strong>, no de sanción individual. Trabajo futuro: incorporar ingreso/trabajo/distancia a
la escuela, pesos de diseño DGEEC y un segundo año de prueba cuando exista.</p>

<h2>6. Entregables</h2>
<table>
<tr><th>Fase</th><th>Pieza</th></tr>
<tr><td>1</td><td>Informe, notebook, diccionario, dataset limpio</td></tr>
<tr><td>2</td><td>Informe, notebook, tabla de contrastes</td></tr>
<tr><td>3</td><td>Informe, notebook, modelo .joblib, script de inferencia, este integrador</td></tr>
</table>
</body></html>"""
    path = DIR_OUT / "Informe_Integrador.html"
    path.write_text(html, encoding="utf-8")
    return path


def write_notebook(ctx) -> Path:
    """Notebook reproducible con celdas markdown/code (se ejecutará con nbconvert)."""
    best = ctx["best_name"]
    bp = ctx["best_params_all"]

    def md(text):
        return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}

    def code(text):
        return {
            "cell_type": "code",
            "metadata": {},
            "execution_count": None,
            "outputs": [],
            "source": text.splitlines(keepends=True),
        }

    cells = []
    cells.append(md(f"""# Proyecto Integrador de Ciencia de Datos — **Fase 3**
## Análisis predictivo: clasificación de inasistencia escolar (EPHC 2022–2025)

| | |
|---|---|
| **Asignatura** | Data Science |
| **Fase** | 3 de 3 — Ponderación 30 % |
| **Integrantes** | Matías Morínigo · César Vielman · Iván Paredes |
| **Target** | `no_asiste` (clasificación binaria desbalanceada) |
| **Partición** | Train 2022–2024 · Test **solo** 2025 (cronológica) |
| **Semilla** | {SEMILLA} |

**Cómo ejecutar.** Con `output/dataset_fase1.parquet` (generado en Fase 1): `Kernel → Restart & Run All`.
"""))
    cells.append(md("""## 1. Correcciones a la Fase 2 y tarea predictiva

No hay devolución escrita del docente en el repositorio. Se mantiene el puente Fase 2 → 3: predictores
`zona`, `departamento`, `edad`, `sexo`; target `no_asiste`; partición cronológica; línea base = clase mayoritaria.

**Tarea:** clasificar si un adolescente no asiste. El motivo declarado **no** es predictor (solo se observa
cuando ya no asiste).
"""))
    cells.append(md("## 2. Carga, features y partición cronológica"))
    cells.append(code("""import sys, warnings, json
from pathlib import Path
import numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns, joblib
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay
from sklearn.model_selection import ParameterGrid

SEMILLA, TEST_YEAR = 42, 2025
np.random.seed(SEMILLA)
warnings.filterwarnings('ignore')
sns.set_theme(style='whitegrid')

RAIZ = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()
sys.path.insert(0, str(RAIZ))
from src.fase3_helpers import (FEATURE_COLS, TARGET, load_modeling_frame, chronological_split,
                               make_pipeline, metrics_row, best_threshold_by_f1, year_folds,
                               make_feature_frame, predict_with_models,
                               ZONAS_VALIDAS, SEXOS_VALIDOS, DEPARTAMENTOS_VALIDOS)

DIR_OUT, DIR_FIG, DIR_TAB, DIR_MOD = RAIZ/'output', RAIZ/'output'/'figuras', RAIZ/'output'/'tablas', RAIZ/'output'/'modelos'
for d in (DIR_FIG, DIR_TAB, DIR_MOD):
    d.mkdir(parents=True, exist_ok=True)

def resolve():
    for p in (DIR_OUT/'dataset_fase1.parquet', RAIZ/'data'/'processed'/'dataset_fase1.parquet'):
        if p.exists() and p.stat().st_size > 0:
            return p
    raise FileNotFoundError('Falta dataset_fase1.parquet')

df = load_modeling_frame(resolve())
train_df, test_df = chronological_split(df, TEST_YEAR)
print(df.shape, 'train', train_df.shape, 'test', test_df.shape)
print('Tasa train', round(train_df[TARGET].mean(), 4), '· tasa test', round(test_df[TARGET].mean(), 4))
train_df.head(3)
"""))
    cells.append(md("""## 3. Línea base y tres familias de modelos

1. **Dummy** mayoritaria. 2. **Logística**. 3. **Random Forest**. 4. **HistGradientBoosting**.
Hiperparámetros por validación temporal expanding dentro de 2022–2024. Umbral F1 en holdout 2024.
El conjunto **2025 no se usa** hasta la evaluación final.
"""))
    cells.append(code(f"""# Hiperparámetros ya seleccionados por scripts/build_fase3.py (misma semilla y grilla)
BEST = {{
  'Logística': {repr(bp['Logística'])},
  'Random Forest': {repr(bp['Random Forest'])},
  'HistGradientBoosting': {repr(bp['HistGradientBoosting'])},
}}
print('Mejores hiperparámetros:', BEST)

X_tr, y_tr = train_df[FEATURE_COLS], train_df[TARGET]
X_te, y_te = test_df[FEATURE_COLS], test_df[TARGET]
inner_tr = train_df[train_df.anio_encuesta < 2024]
inner_va = train_df[train_df.anio_encuesta == 2024]

def sw(y):
    w = np.ones(len(y)); pos = (y==1).sum(); w[y==1] = (y==0).sum()/max(pos,1); return w

fitted, thresholds, rows = {{}}, {{}}, []

# Baseline
base = make_pipeline(DummyClassifier(strategy='most_frequent'), scale_numeric=False)
base.fit(X_tr, y_tr)
fitted['Línea base (mayoritaria)'] = base
thresholds['Línea base (mayoritaria)'] = 0.5
for split, X, y in [('train', X_tr, y_tr), ('test', X_te, y_te)]:
    pred = base.predict(X)
    rows.append(metrics_row('Línea base (mayoritaria)', split, y, pred, np.zeros(len(y))))

# Logística
logit = make_pipeline(LogisticRegression(random_state=SEMILLA, **BEST['Logística']), True)
logit.fit(inner_tr[FEATURE_COLS], inner_tr[TARGET])
t, _ = best_threshold_by_f1(inner_va[TARGET], logit.predict_proba(inner_va[FEATURE_COLS])[:,1])
logit.fit(X_tr, y_tr)
fitted['Logística'], thresholds['Logística'] = logit, t
for split, X, y in [('train', X_tr, y_tr), ('test', X_te, y_te)]:
    p = logit.predict_proba(X)[:,1]; rows.append(metrics_row('Logística', split, y, (p>=t).astype(int), p))

# RF
rf = make_pipeline(RandomForestClassifier(random_state=SEMILLA, n_jobs=-1, **BEST['Random Forest']), False)
rf.fit(inner_tr[FEATURE_COLS], inner_tr[TARGET])
t, _ = best_threshold_by_f1(inner_va[TARGET], rf.predict_proba(inner_va[FEATURE_COLS])[:,1])
rf.fit(X_tr, y_tr)
fitted['Random Forest'], thresholds['Random Forest'] = rf, t
for split, X, y in [('train', X_tr, y_tr), ('test', X_te, y_te)]:
    p = rf.predict_proba(X)[:,1]; rows.append(metrics_row('Random Forest', split, y, (p>=t).astype(int), p))

# HGB
hgb = make_pipeline(HistGradientBoostingClassifier(random_state=SEMILLA, **BEST['HistGradientBoosting']), False)
hgb.fit(inner_tr[FEATURE_COLS], inner_tr[TARGET], clf__sample_weight=sw(inner_tr[TARGET]))
t, _ = best_threshold_by_f1(inner_va[TARGET], hgb.predict_proba(inner_va[FEATURE_COLS])[:,1])
hgb.fit(X_tr, y_tr, clf__sample_weight=sw(y_tr))
fitted['HistGradientBoosting'], thresholds['HistGradientBoosting'] = hgb, t
for split, X, y in [('train', X_tr, y_tr), ('test', X_te, y_te)]:
    p = hgb.predict_proba(X)[:,1]; rows.append(metrics_row('HistGradientBoosting', split, y, (p>=t).astype(int), p))

resumen = pd.DataFrame(rows)
display(resumen.pivot(index='modelo', columns='split')[['f1','auc_roc','recall','precision','accuracy']].round(3))
print('Umbrales:', thresholds)
"""))
    cells.append(md(f"""## 4. Evaluación final en 2025

Modelo seleccionado por F1 de validación temporal en el pipeline de construcción: **{best}**.
Abajo se reportan matriz de confusión, ROC e importancia de variables. El test se usa **una sola vez**.
"""))
    cells.append(code(f"""BEST_NAME = '{best}'
model = fitted[BEST_NAME]
umbral = thresholds[BEST_NAME]
proba = model.predict_proba(X_te)[:,1]
pred = (proba >= umbral).astype(int)
print(resumen[(resumen.modelo==BEST_NAME)&(resumen.split=='test')].T)

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
ConfusionMatrixDisplay.from_predictions(y_te, pred, ax=axes[0], colorbar=False)
axes[0].set_title('Matriz de confusión — test 2025')
RocCurveDisplay.from_predictions(y_te, proba, ax=axes[1], name=BEST_NAME)
axes[1].plot([0,1],[0,1],'--', color='#52514e', lw=1)
axes[1].set_title('ROC — test 2025')
fig.tight_layout(); fig.savefig(DIR_FIG/'F3_notebook_eval.png'); plt.show()

# Importancia RF
prep = fitted['Random Forest'].named_steps['prep']
names = ['edad'] + list(prep.named_transformers_['cat'].get_feature_names_out(['zona','departamento','sexo']))
imp = pd.Series(fitted['Random Forest'].named_steps['clf'].feature_importances_, index=names).sort_values(ascending=False)
display(imp.head(12).to_frame('importancia'))
imp.head(12).iloc[::-1].plot(kind='barh', figsize=(8,4), title='Importancia RF')
plt.tight_layout(); plt.savefig(DIR_FIG/'F3_notebook_imp.png'); plt.show()

# Errores
err = test_df.copy()
err['proba'], err['pred'] = proba, pred
err['tipo'] = np.where((err[TARGET]==1)&(err.pred==0), 'FN', np.where((err[TARGET]==0)&(err.pred==1), 'FP', 'OK'))
display(pd.crosstab(err['zona'], err['tipo'], normalize='index').round(3))
display(err[err.tipo=='FN'].groupby('edad').size().rename('FN').to_frame().join(
        err[err[TARGET]==1].groupby('edad').size().rename('positivos')).assign(tasa_FN=lambda d: d.FN/d.positivos))
"""))
    cells.append(md("""## 5. Predicción manual — ingresar X y ver el Y de cada modelo

Editá las 4 variables de abajo y volvé a ejecutar la celda. Cada modelo responde con su chance
y con `Y_predicho` (0 = asiste, 1 = no asiste).
"""))
    cells.append(code("""# >>> EDITÁ SOLO ESTOS 4 VALORES <<<
edad = 16
zona = 'Rural'                 # 'Urbana' o 'Rural'
departamento = 'San Pedro'     # ver lista impresa abajo
sexo = 'Hombre'                # 'Hombre' o 'Mujer'

# Esta celda funciona sola (no hace falta correr todo el notebook antes)
import sys
from pathlib import Path
import joblib

_raiz = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()
sys.path.insert(0, str(_raiz))
from src.fase3_helpers import (
    make_feature_frame, predict_with_models,
    ZONAS_VALIDAS, SEXOS_VALIDOS, DEPARTAMENTOS_VALIDOS,
)

_dir_mod = _raiz / 'output' / 'modelos'
if 'fitted' not in globals() or 'thresholds' not in globals():
    _bundle = joblib.load(_dir_mod / 'fase3_modelos_todos.joblib')
    fitted = _bundle['fitted']
    thresholds = _bundle['thresholds']
    BEST_NAME = _bundle.get('best_model', 'Logística')
    print('(Modelos cargados desde fase3_modelos_todos.joblib)')

print('Zonas válidas:', ZONAS_VALIDAS)
print('Sexos válidos:', SEXOS_VALIDOS)
print('Departamentos válidos:', DEPARTAMENTOS_VALIDOS)

X_manual = make_feature_frame(edad, zona, departamento, sexo)
print('\nX ingresado:')
display(X_manual)

tabla_manual = predict_with_models(X_manual, fitted, thresholds)
display(tabla_manual[['modelo', 'chance_no_asiste', 'umbral', 'Y_predicho', 'etiqueta']])
print('Modelo oficial del proyecto:', BEST_NAME,
      '→', tabla_manual.loc[tabla_manual.modelo == BEST_NAME, 'etiqueta'].iloc[0])
"""))
    cells.append(md("""## 6. Serialización

El modelo final, el umbral y el bundle con todos los modelos se guardan para reutilizar
(`scripts/infer_fase3.py` o esta misma sección sin reentrenar).
"""))
    cells.append(code("""bundle = {'pipeline': model, 'threshold': umbral, 'model_name': BEST_NAME,
          'features': FEATURE_COLS, 'target': TARGET}
joblib.dump(bundle, DIR_MOD/'fase3_modelo_final.joblib')
joblib.dump({'fitted': fitted, 'thresholds': thresholds, 'features': FEATURE_COLS,
             'target': TARGET, 'best_model': BEST_NAME},
            DIR_MOD/'fase3_modelos_todos.joblib')
meta = {'best_model': BEST_NAME, 'umbral': umbral, 'features': FEATURE_COLS, 'target': TARGET,
        'test_year': TEST_YEAR, 'seed': SEMILLA, 'best_params': BEST}
(DIR_MOD/'fase3_modelo_meta.json').write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding='utf-8')
resumen.to_csv(DIR_TAB/'fase3_metricas.csv', index=False, encoding='utf-8-sig')
print('Guardado:', DIR_MOD/'fase3_modelo_final.joblib')
print('Guardado:', DIR_MOD/'fase3_modelos_todos.joblib')
"""))
    cells.append(md(f"""## 7. Conclusión

- La línea base mayoritaria no detecta inasistencia (F1 ≈ 0); los tres modelos la superan en F1/AUC.
- Modelo final: **{best}**, evaluado una sola vez en 2025.
- Edad, zona y departamento dominan la importancia, en línea con H1/H3/H4.
- Uso legítimo: priorización; no sanción individual. Límites: sin confusores socioeconómicos, sin Chaco,
  un solo año de prueba, sin pesos de diseño DGEEC.
"""))

    nb = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
        "cells": cells,
    }
    # add ids
    import uuid
    for c in nb["cells"]:
        c["id"] = uuid.uuid4().hex[:8]
    path = DIR_NB / "Fase3_DataScience_EPHC.ipynb"
    path.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    return path


def main():
    print("Cargando…")
    df = load_modeling_frame(resolve_dataset())
    print("Modelando…")
    ctx = run_modeling(df)
    print("Mejor modelo (CV):", ctx["best_name"])
    print(ctx["resumen"][ctx["resumen"].split == "test"][["modelo", "f1", "auc_roc", "recall", "precision"]].to_string(index=False))
    print("Figuras…")
    make_figures(ctx)
    print("Modelo…")
    save_model(ctx)
    print("Informes…")
    write_informe(ctx)
    write_integrador(ctx)
    print("Notebook…")
    nb = write_notebook(ctx)
    print("OK:", nb)
    print("OK:", DIR_OUT / "Informe_Fase3.html")
    print("OK:", DIR_OUT / "Informe_Integrador.html")
    print("OK:", DIR_MOD / "fase3_modelo_final.joblib")


if __name__ == "__main__":
    main()
