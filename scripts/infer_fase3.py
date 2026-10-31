#!/usr/bin/env python3
"""Aplica modelos de Fase 3 a datos nuevos o a un caso manual.

Uso con archivo:
  python scripts/infer_fase3.py datos.csv --salida predicciones.csv

Uso manual (un adolescente):
  python scripts/infer_fase3.py --edad 16 --zona Rural --departamento "San Pedro" --sexo Hombre
  python scripts/infer_fase3.py --edad 16 --zona Rural --departamento "San Pedro" --sexo Hombre --todos
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.fase3_helpers import (  # noqa: E402
    FEATURE_COLS,
    make_feature_frame,
    predict_with_models,
)

DEFAULT_MODEL = ROOT / "output" / "modelos" / "fase3_modelo_final.joblib"
DEFAULT_ALL = ROOT / "output" / "modelos" / "fase3_modelos_todos.joblib"
DEFAULT_META = ROOT / "output" / "modelos" / "fase3_modelo_meta.json"


def load_frame(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Inferencia Fase 3 — inasistencia escolar")
    parser.add_argument("entrada", type=Path, nargs="?", default=None, help="CSV o parquet con las features")
    parser.add_argument("--modelo", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--todos-modelos", type=Path, default=DEFAULT_ALL)
    parser.add_argument("--meta", type=Path, default=DEFAULT_META)
    parser.add_argument("--salida", type=Path, default=None, help="CSV de salida (default: stdout)")
    parser.add_argument("--umbral", type=float, default=None)
    parser.add_argument("--edad", type=float, default=None)
    parser.add_argument("--zona", type=str, default=None)
    parser.add_argument("--departamento", type=str, default=None)
    parser.add_argument("--sexo", type=str, default=None)
    parser.add_argument(
        "--todos",
        action="store_true",
        help="Comparar todos los modelos (requiere fase3_modelos_todos.joblib)",
    )
    args = parser.parse_args(argv)

    manual = all(v is not None for v in (args.edad, args.zona, args.departamento, args.sexo))
    if args.entrada is None and not manual:
        parser.error("Pasá un archivo de entrada o los 4 flags --edad --zona --departamento --sexo")

    meta = {}
    if args.meta.exists():
        meta = json.loads(args.meta.read_text(encoding="utf-8"))

    if args.todos:
        if not args.todos_modelos.exists():
            print(
                f"No está el bundle de todos los modelos: {args.todos_modelos}\n"
                "Corré la celda de serialización del notebook o scripts/build_fase3.py",
                file=sys.stderr,
            )
            return 1
        bundle_all = joblib.load(args.todos_modelos)
        fitted = bundle_all["fitted"]
        thresholds = dict(bundle_all["thresholds"])
        if args.umbral is not None:
            for k in list(thresholds):
                thresholds[k] = args.umbral

        if manual:
            frames = [make_feature_frame(args.edad, args.zona, args.departamento, args.sexo)]
        else:
            df = load_frame(args.entrada)
            faltan = [c for c in FEATURE_COLS if c not in df.columns]
            if faltan:
                print(f"Faltan columnas: {faltan}", file=sys.stderr)
                return 1
            frames = [
                make_feature_frame(
                    row["edad"], row["zona"], row["departamento"], row["sexo"]
                )
                for _, row in df.iterrows()
            ]

        outs = []
        for i, Xi in enumerate(frames):
            t = predict_with_models(Xi, fitted, thresholds)
            t.insert(0, "fila", i)
            outs.append(t)
        out = pd.concat(outs, ignore_index=True)
        if args.salida:
            args.salida.parent.mkdir(parents=True, exist_ok=True)
            out.to_csv(args.salida, index=False, encoding="utf-8-sig")
            print(f"Guardado: {args.salida}  ({len(out)} filas)")
        else:
            cols = ["fila", "modelo", "chance_no_asiste", "umbral", "Y_predicho", "etiqueta"]
            if manual:
                print("X:", frames[0].to_dict(orient="records")[0])
                print(out[cols].to_string(index=False))
            else:
                out.to_csv(sys.stdout, index=False)
        return 0

    if not args.modelo.exists():
        print(f"No está el modelo: {args.modelo}", file=sys.stderr)
        return 1

    bundle = joblib.load(args.modelo)
    pipe = bundle["pipeline"] if isinstance(bundle, dict) else bundle
    features = meta.get("features", FEATURE_COLS)
    if args.umbral is not None:
        umbral = float(args.umbral)
    elif isinstance(bundle, dict) and "threshold" in bundle:
        umbral = float(bundle["threshold"])
    else:
        umbral = float(meta.get("umbral", 0.5))

    if manual:
        base = make_feature_frame(args.edad, args.zona, args.departamento, args.sexo)
        X = base[features]
    else:
        base = load_frame(args.entrada)
        faltan = [c for c in features if c not in base.columns]
        if faltan:
            print(f"Faltan columnas: {faltan}", file=sys.stderr)
            return 1
        X = base[features].copy()
        X["edad"] = X["edad"].astype(float)
        for c in ["zona", "departamento", "sexo"]:
            X[c] = X[c].astype(str)

    proba = pipe.predict_proba(X)[:, 1]
    pred = (proba >= umbral).astype(int)
    out = base.copy()
    out["proba_no_asiste"] = proba
    out["pred_no_asiste"] = pred
    out["umbral_usado"] = umbral
    out["etiqueta"] = ["no asiste" if p == 1 else "asiste" for p in pred]

    if args.salida:
        args.salida.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(args.salida, index=False, encoding="utf-8-sig")
        print(f"Guardado: {args.salida}  ({len(out)} filas, umbral={umbral})")
    elif manual:
        print(
            out[
                [
                    "edad",
                    "zona",
                    "departamento",
                    "sexo",
                    "proba_no_asiste",
                    "pred_no_asiste",
                    "etiqueta",
                ]
            ].to_string(index=False)
        )
    else:
        out.to_csv(sys.stdout, index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
