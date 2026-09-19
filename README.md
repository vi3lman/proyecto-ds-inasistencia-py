# Inasistencia escolar en la adolescencia paraguaya: magnitud, motivos declarados y brechas territoriales (2022–2024)

Proyecto Integrador de Ciencia de Datos — **Fase 1**: comprensión del problema, comprensión y selección de
los datos, limpieza y transformación, análisis univariado y bivariado, análisis descriptivo y exploratorio.

**Integrantes:** Matías Morínigo · César Vielman · Iván Paredes

**Entrega Fase 1:** `output/Informe_Fase1.pdf` · `output/fase1_comprension_problema.pdf` · notebook ejecutado.

---

## Pregunta de investigación

> ¿En qué medida la zona de residencia (urbana/rural) y el departamento se asocian con la tasa de
> inasistencia escolar de los adolescentes de 12 a 17 años en Paraguay, y qué motivos declaran los hogares
> para explicar esa inasistencia, entre 2022 y 2024?

## Por qué la fuente es la EPHC y no el MEC

La propuesta original de este proyecto apuntaba al abandono escolar usando registros administrativos del
MEC. Al inventariar los archivos efectivamente disponibles se comprobó que **ningún dataset del MEC
contiene una variable de abandono o deserción**. La
**Encuesta Permanente de Hogares Continua (EPHC)** de la DGEEC, en cambio, pregunta directamente al hogar
si el adolescente asiste y, si no, por qué — la variable que la propuesta original necesitaba. El costo de
este cambio: la unidad de análisis pasa de *establecimiento* a *persona dentro de un hogar*, con diseño
muestral complejo que exige ponderar por el factor de expansión (`FEX`) en todo cálculo. El detalle completo
de esta decisión está documentado en la sección "Nota metodológica" del notebook.

## Estructura del repositorio

```
.
├── data/
│   ├── raw/            # REG02_EPHC_ANUAL_{2022,2023,2024}.csv, SIN modificar (no versionados)
│   └── processed/      # dataset_fase1.parquet — generado por el notebook
├── notebooks/
│   └── Fase1_DataScience_EPHC.ipynb
├── src/                # funciones reutilizables extraídas del notebook (Fases 2 y 3)
├── output/
│   ├── figuras/        # G1…G8 en .png
│   └── tablas/         # diccionarios de datos, bitácora de limpieza, series
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

# 1. Descargar los tres archivos REG02_EPHC_ANUAL_*.csv desde la DGEEC
#    y colocarlos SIN MODIFICAR en data/raw/
# 2. Ejecutar el notebook completo
jupyter nbconvert --to notebook --execute --inplace notebooks/Fase1_DataScience_EPHC.ipynb
jupyter nbconvert --to html notebooks/Fase1_DataScience_EPHC.ipynb --output ../output/Fase1.html
```

El notebook se ejecuta de principio a fin sin intervención manual y regenera
`data/processed/dataset_fase1.parquet` a partir de los archivos originales.

> **Detalle de formato crítico:** los archivos EPHC usan `;` como separador de columnas y `,` como separador
> **decimal**. Si se lee sin `decimal=','`, el factor de expansión (`FEX.2022`) se interpreta como texto y
> cualquier suma posterior concatena strings en vez de sumar — sin ningún error visible. El notebook ya
> aplica esta corrección; queda documentado para quien reutilice el código en otro contexto.

## Fuente de datos

| Ítem | Valor |
|---|---|
| Fuente | Dirección General de Estadística, Encuestas y Censos (DGEEC), Paraguay |
| Encuesta | Encuesta Permanente de Hogares Continua (EPHC), rondas 2022, 2023 y 2024 |
| Archivos | `REG02_EPHC_ANUAL_2022.csv`, `REG02_EPHC_ANUAL_2023.csv`, `REG02_EPHC_ANUAL_2024.csv` |
| Diccionario de variables | `diccionario_EPHC_ANUAL_2023.xls`, hoja "EPHC 2023" |
| Unidad de análisis | persona dentro de un hogar, año de encuesta (corte transversal repetido, no panel) |
| Licencia | Microdatos de uso público de la DGEEC |

## Limitación de cobertura verificada empíricamente

Los departamentos de **Boquerón y Alto Paraguay** (región del Chaco) no aparecen en la muestra de la EPHC en
ninguno de los tres años relevados. Es un límite sistemático de la fuente, verificado en la Etapa 2.6 del
notebook: ningún resultado de este proyecto puede extenderse a esa región.

## Reproducibilidad

- Rutas relativas en todo el código.
- Semilla aleatoria fijada (`SEMILLA = 42`).
- Todas las decisiones de limpieza quedan en `output/tablas/bitacora_limpieza.csv`.
- Todo estadístico descriptivo está ponderado por el factor de expansión muestral (`factor_expansion`).

## Calendario

| Fase | Contenido | Entrega |
|---|---|---|
| 1 | Comprensión, preparación y análisis exploratorio | lunes 15 de septiembre |
| 2 | Análisis inferencial | viernes 3 de octubre |
| 3 | Análisis predictivo | viernes 24 de octubre |
