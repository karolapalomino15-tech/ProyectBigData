# Muestras de datos por etapa

Muestras pequeñas (50 a 60 filas) para mostrar el dato en cada etapa sin subir los cientos de MB del dataset.
Se generan con `src/06_generar_muestras.py` después de ejecutar el pipeline. Las etapas 00 a 03 siguen las mismas
cinco simulaciones (clases 0, 1, 4, 6 y 15), primeros 10 pasos de tiempo, para poder seguir un registro de punta a punta.

| Carpeta | Archivo | Contenido |
|---|---|---|
| `00_crudo` | `muestra_cruda.csv` | 50 filas tomadas directamente del pickle: `sample_id`, `label` (0 = normal, 1 a 28 = falla), `fold`, `t` y las 34 variables `v00` a `v33` |
| `01_bronze` | `muestra_bronze.csv` | Las mismas 50 filas leídas desde el Parquet de bronze (idénticas al dato crudo) |
| `02_silver` | `muestra_silver.csv` | Las mismas filas en silver: sin las variables constantes `v26`, `v30` y `v33` (31 variables útiles) y con la columna `es_falla` |
| `03_gold` | `muestra_gold_deteccion.csv` | 58 filas: dos simulaciones por clase con el resultado por simulación (`detectada_pca`, `retardo_pca`, `frac_excede`, `score_if`, `detectada_if`) |
| `03_gold` | `muestra_gold_t2_spe_por_paso.csv` | 50 filas con T², SPE y si el paso excede el umbral, calculados con el modelo del fold correspondiente |
| `04_streaming` | `muestra_microlote.csv` | 60 filas del primer micro-lote emitido en la simulación de streaming |
| `04_streaming` | `streaming_bitacora.json` | Alarmas generadas en la simulación (paso de alarma por simulación) |
