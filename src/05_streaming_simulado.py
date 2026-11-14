"""Simulación de flujo en tiempo real con Spark Structured Streaming.
Se 'emiten' ventanas del fold 1 como micro-lotes de PASOS_POR_LOTE pasos (archivos parquet que van
llegando a data/stream_in). Cada micro-lote se puntúa con el modelo PCA entrenado SOLO con datos
normales de los otros folds (modelo_fold1.npz) y se mantiene el contador de pasos consecutivos
sobre el umbral entre micro-lotes."""
import json, shutil, threading, time
import numpy as np, pandas as pd
import pyarrow.dataset as ds
from pyspark.sql import functions as F
from config import get_spark, SILVER, GOLD, SALIDAS, RAIZ, K_CONSECUTIVAS

ENTRADA = RAIZ / "data" / "stream_in"
PASOS_POR_LOTE = 50
EJEMPLOS = [0, 1, 4, 6]          # clase normal + 3 fallas

def preparar_muestras(cols):
    tab = ds.dataset(str(SILVER), format="parquet", partitioning="hive").to_table(
        filter=(ds.field("fold") == 1) & ds.field("label").isin(EJEMPLOS),
        columns=["sample_id", "label", "t"] + cols).to_pandas()
    ids = [int(tab[tab.label == l].sample_id.iloc[0]) for l in EJEMPLOS]
    return tab[tab.sample_id.isin(ids)].sort_values(["sample_id", "t"])

def emisor(tab):
    """Productor: deposita un archivo por micro-lote, intercalando las simulaciones."""
    for ini in range(0, 600, PASOS_POR_LOTE):
        lote = tab[(tab.t >= ini) & (tab.t < ini + PASOS_POR_LOTE)]
        lote.to_parquet(ENTRADA / f"lote_{ini:04d}.parquet")
        time.sleep(1.0)

def main():
    mod = np.load(GOLD / "modelo_fold1.npz", allow_pickle=True)
    cols = list(mod["cols"]); mu, sd, P, lam = mod["mu"], mod["sd"], mod["P"], mod["lam"]
    thr_t2, thr_spe = float(mod["thr_t2"]), float(mod["thr_spe"])
    tab = preparar_muestras(cols)
    etiqueta = tab.groupby("sample_id").label.first().to_dict()

    if ENTRADA.exists():
        shutil.rmtree(ENTRADA)
    ENTRADA.mkdir(parents=True)
    spark = get_spark("TEP-streaming")
    esquema = spark.read.parquet(str(SILVER)).select("sample_id", "label", "t", *cols).schema

    consec = {s: 0 for s in etiqueta}        # estado: pasos consecutivos sobre umbral
    primera = {}                              # primera alarma: sample_id -> (paso, micro-lote)
    bitacora = []

    def procesar(df, batch_id):
        pdf = df.toPandas().sort_values(["sample_id", "t"])
        if pdf.empty:
            return
        X = (pdf[cols].to_numpy() - mu) / sd
        T = X @ P
        pdf["t2"] = (T ** 2 / lam).sum(1); pdf["spe"] = (X ** 2).sum(1) - (T ** 2).sum(1)
        for sid, g in pdf.groupby("sample_id"):
            for t, a, b in zip(g.t, g.t2, g.spe):
                consec[sid] = consec[sid] + 1 if (a > thr_t2 or b > thr_spe) else 0
                if consec[sid] >= K_CONSECUTIVAS and sid not in primera:
                    primera[sid] = (int(t), int(batch_id))
                    bitacora.append({"microlote": int(batch_id), "sample_id": int(sid),
                                     "clase": int(etiqueta[sid]), "paso_alarma": int(t)})
                    print(f"[microlote {batch_id}] ALARMA simulación {sid} (clase {etiqueta[sid]}) en el paso {t}", flush=True)

    flujo = (spark.readStream.schema(esquema).option("maxFilesPerTrigger", 1).parquet(str(ENTRADA)))
    q = flujo.writeStream.foreachBatch(procesar).trigger(processingTime="1 second").start()

    hilo = threading.Thread(target=emisor, args=(tab,)); hilo.start(); hilo.join()
    time.sleep(6)
    q.stop(); spark.stop()

    sin_alarma = [{"sample_id": int(s), "clase": int(etiqueta[s])} for s in etiqueta if s not in primera]
    resumen = {"alarmas": bitacora, "sin_alarma": sin_alarma, "pasos_por_microlote": PASOS_POR_LOTE}
    (SALIDAS / "tablas" / "streaming_bitacora.json").write_text(json.dumps(resumen, indent=2))
    print("Sin alarma:", sin_alarma)

if __name__ == "__main__":
    main()
