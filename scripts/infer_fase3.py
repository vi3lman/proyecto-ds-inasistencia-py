#!/usr/bin/env python3
"""Aplica el modelo final de Fase 3 a datos nuevos.

Uso:
  python scripts/infer_fase3.py datos.csv --salida predicciones.csv
  python scripts/infer_fase3.py datos.parquet

El CSV/parquet de entrada debe incluir al menos: edad, sexo, zona, departamento.
Opcional: umbral distinto del guardado con --umbral 0.3
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = ROOT / "output" / "modelos" / "fase3_modelo_final.joblib"
DEFAULT_META = ROOT / "output" / "modelos" / "fase3_modelo_meta.json"


def load_frame(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Inferencia Fase 3 — inasistencia escolar")
    parser.add_argument("entrada", type=Path, help="CSV o parquet con las features")
    parser.add_argument("--modelo", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--meta", type=Path, default=DEFAULT_META)
    parser.add_argument("--salida", type=Path, default=None, help="CSV de salida (default: stdout)")
    parser.add_argument("--umbral", type=float, default=None)
    args = parser.parse_args(argv)

    if not args.modelo.exists():
        print(f"No está el modelo: {args.modelo}", file=sys.stderr)
        return 1

    bundle = joblib.load(args.modelo)
    pipe = bundle["pipeline"] if isinstance(bundle, dict) else bundle
    meta = {}
    if args.meta.exists():
        meta = json.loads(args.meta.read_text(encoding="utf-8"))
    features = meta.get("features", ["edad", "zona", "departamento", "sexo"])
    umbral = args.umbral if args.umbral is not None else float(meta.get("umbral", 0.5))

    df = load_frame(args.entrada)
    faltan = [c for c in features if c not in df.columns]
    if faltan:
        print(f"Faltan columnas: {faltan}", file=sys.stderr)
        return 1

    X = df[features].copy()
    X["edad"] = X["edad"].astype(float)
    for c in ["zona", "departamento", "sexo"]:
        if c in X.columns:
            X[c] = X[c].astype(str)

    proba = pipe.predict_proba(X)[:, 1]
    pred = (proba >= umbral).astype(int)
    out = df.copy()
    out["proba_no_asiste"] = proba
    out["pred_no_asiste"] = pred
    out["umbral_usado"] = umbral

    if args.salida:
        args.salida.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(args.salida, index=False, encoding="utf-8-sig")
        print(f"Guardado: {args.salida}  ({len(out)} filas, umbral={umbral})")
    else:
        out.to_csv(sys.stdout, index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
