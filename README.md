# Inasistencia escolar en la adolescencia paraguaya: magnitud, motivos declarados y brechas territoriales (2022–2025)

Proyecto Integrador de Ciencia de Datos — **Fase 1** (descriptivo) y **Fase 2** (inferencial).

**Integrantes:** Matías Morínigo · César Vielman · Iván Paredes

**Entrega Fase 2:** `output/Informe_Fase2.pdf` · `notebooks/Fase2_DataScience_EPHC.ipynb` · `output/Fase2.html` · `output/tablas/fase2_resumen_contrastes.csv`

---

## Pregunta de investigación

> ¿En qué medida la zona de residencia (urbana/rural) y el departamento se asocian con la tasa de
> inasistencia escolar de los adolescentes de 12 a 17 años en Paraguay, y qué motivos declaran los hogares
> para explicar esa inasistencia, entre 2022 y 2025?

## Por qué la fuente es la EPHC

El proyecto mide la inasistencia escolar actual de adolescentes de 12 a 17 años y los motivos que declara
el hogar. La **Encuesta Permanente de Hogares Continua (EPHC)** de la DGEEC pregunta cada año si la persona
asiste (ED08) y, si no, por qué (ED10). La unidad de análisis es la *persona dentro de un hogar*; el diseño
muestral exige ponderar por el factor de expansión (`FEX`) en todo cálculo descriptivo. La inferencia de la
Fase 2 se realiza sobre el *n* muestral (sin pesos de replicación).

## Estructura del repositorio

```
.
├── data/
│   └── raw/            # REG02_EPHC_ANUAL_{2022,2023,2024,2025}.csv (no versionados)
├── notebooks/
│   ├── Fase1_DataScience_EPHC.ipynb
│   └── Fase2_DataScience_EPHC.ipynb
├── src/
│   └── fase2_helpers.py
├── scripts/
│   └── build_fase2.py  # regenera tablas, figuras, notebook e informe de Fase 2
├── output/
│   ├── dataset_fase1.parquet
│   ├── Informe_Fase2.pdf / .html
│   ├── Fase2.html
│   ├── figuras/        # G1…G8 (Fase 1) y F2_* (Fase 2)
│   └── tablas/         # diccionarios, bitácora, fase2_resumen_contrastes.csv
├── requirements.txt
└── README.md
```

## Cómo reproducir el trabajo

```bash
git clone <url-del-repositorio>
cd proyecto-ds-inasistencia-py

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Datos: REG02_EPHC_ANUAL_{2022,2023,2024,2025}.csv en data/raw/
# (en 2025 el peso se llama FACTOR; el notebook lo unifica con FEX.2022)

# Fase 1
jupyter nbconvert --to notebook --execute --inplace notebooks/Fase1_DataScience_EPHC.ipynb
jupyter nbconvert --to html notebooks/Fase1_DataScience_EPHC.ipynb --output ../output/Fase1.html

# Fase 2 (requiere output/dataset_fase1.parquet)
python scripts/build_fase2.py
jupyter nbconvert --to notebook --execute --inplace notebooks/Fase2_DataScience_EPHC.ipynb
jupyter nbconvert --to html notebooks/Fase2_DataScience_EPHC.ipynb --output-dir output --output Fase2.html
```

> **Detalle de formato crítico:** los archivos EPHC usan `;` como separador de columnas y `,` como separador
> **decimal**. Si se lee sin `decimal=','`, el factor de expansión se interpreta como texto.

## Fuente de datos

| Ítem | Valor |
|---|---|
| Fuente | Dirección General de Estadística, Encuestas y Censos (DGEEC), Paraguay |
| Encuesta | Encuesta Permanente de Hogares Continua (EPHC), rondas 2022–2025 |
| Archivos | `REG02_EPHC_ANUAL_2022.csv` … `REG02_EPHC_ANUAL_2025.csv` |
| Unidad de análisis | persona dentro de un hogar, año de encuesta |
| Licencia | Microdatos de uso público de la DGEEC |

## Limitación de cobertura

Los departamentos de **Boquerón y Alto Paraguay** no aparecen en la muestra EPHC de los años relevados.
Ningún resultado se extiende a esa región.

## Reproducibilidad

- Rutas relativas; semilla `SEMILLA = 42`.
- Fase 1: descriptivos ponderados por `factor_expansion`.
- Fase 2: α = 0.05 a priori; BH sobre H1–H4 y H6; IC bootstrap de la diferencia rural−urbana; helpers en `src/fase2_helpers.py`.

## Calendario

| Fase | Contenido | Entrega |
|---|---|---|
| 1 | Comprensión, preparación y análisis exploratorio | lunes 15 de septiembre |
| 2 | Análisis inferencial | viernes 3 de octubre |
| 3 | Análisis predictivo | viernes 24 de octubre |
