#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/src"
python 01_landing_a_bronze.py      # pickle -> parquet (bronze)
python 02_silver.py                # calidad + variables constantes (silver)
python 03_gold_deteccion.py        # PCA T2/SPE + Isolation Forest, 5 folds (gold)
python 04_reportes.py              # tablas y figuras
python 05_streaming_simulado.py    # simulación Structured Streaming
