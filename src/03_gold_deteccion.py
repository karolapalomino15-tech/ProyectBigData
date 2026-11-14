"""Silver -> Gold (Spark): detección de anomalías con validación cruzada de 5 folds.

Para cada fold de prueba:
  * Entrenamiento = simulaciones NORMALES (label 0) de los otros 4 folds.
  * Estandarización (media/desv. calculadas en Spark sobre datos normales de entrenamiento).
  * Modelo A (flujo / fila a fila): PCA de Spark ML + estadísticos T2 y SPE por paso de tiempo;
    alarma si K pasos consecutivos superan el umbral (percentil de datos normales de entrenamiento).
  * Modelo B (lote / por ventana): Isolation Forest sobre estadísticos por simulación (media y desv.
    de cada variable, agregados en Spark).
Salidas en data/gold y outputs/tablas."""
import time
import numpy as np
import pandas as pd
from pyspark.sql import functions as F, Window
from pyspark.sql.functions import pandas_udf
from pyspark.ml.feature import PCA, VectorAssembler
from pyspark.ml.functions import array_to_vector
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from config import (get_spark, SILVER, GOLD, SALIDAS, PERCENTIL_UMBRAL,
                    K_CONSECUTIVAS, VARIANZA_PCA)

def main():
    t0 = time.time()
    spark = get_spark("TEP-gold")
    df = spark.read.parquet(str(SILVER))
    utiles = [c for c in df.columns if c.startswith("v")]
    p = len(utiles)

    GOLD.mkdir(parents=True, exist_ok=True)
    (SALIDAS / "tablas").mkdir(parents=True, exist_ok=True)
    res_pca, res_if, info_folds = [], [], []

    for fold in range(1, 6):
        train = df.filter((F.col("fold") != fold) & (F.col("label") == 0))
        test = df.filter(F.col("fold") == fold)

        # --- estandarización con datos normales de entrenamiento (Spark) ---
        st = train.select(*[F.mean(c).alias("m_" + c) for c in utiles],
                          *[F.stddev(c).alias("s_" + c) for c in utiles]).first().asDict()
        mu = {c: st["m_" + c] for c in utiles}
        sd = {c: (st["s_" + c] or 1.0) for c in utiles}
        z = lambda d: d.withColumn("z", F.array(*[((F.col(c) - mu[c]) / sd[c]) for c in utiles]))
        tr_z, te_z = z(train), z(test)

        # --- PCA (Spark ML) ---
        tr_vec = tr_z.withColumn("features", array_to_vector("z")).select("features")
        pca = PCA(k=p, inputCol="features", outputCol="pcs").fit(tr_vec)
        ev = np.array(pca.explainedVariance)
        k = int(np.searchsorted(np.cumsum(ev), VARIANZA_PCA) + 1)
        P = np.array(pca.pc.toArray())[:, :k]                 # (p, k)
        lam = ev[:k] * p                                      # varianza total = p tras estandarizar
        # (la varianza de los puntajes = valor propio; total estandarizado = p)

        @pandas_udf("t2 double, spe double")
        def puntuar(arr: pd.Series) -> pd.DataFrame:
            X = np.stack(arr.to_numpy())
            T = X @ P
            return pd.DataFrame({"t2": (T ** 2 / lam).sum(1),
                                 "spe": (X ** 2).sum(1) - (T ** 2).sum(1)})

        def con_scores(d):
            return d.withColumn("sc", puntuar("z")).select(
                "sample_id", "label", "t", "sc.t2", "sc.spe")

        sc_tr = con_scores(tr_z)
        thr_t2, thr_spe = sc_tr.approxQuantile(["t2", "spe"], [PERCENTIL_UMBRAL] * 2, 1e-4)
        thr_t2, thr_spe = thr_t2[0] if isinstance(thr_t2, list) else thr_t2, \
                          thr_spe[0] if isinstance(thr_spe, list) else thr_spe

        sc_te = con_scores(te_z).withColumn(
            "excede", ((F.col("t2") > thr_t2) | (F.col("spe") > thr_spe)).cast("int"))
        w = Window.partitionBy("sample_id").orderBy("t").rowsBetween(-(K_CONSECUTIVAS - 1), 0)
        sc_te = sc_te.withColumn("alarma", (F.sum("excede").over(w) == K_CONSECUTIVAS).cast("int"))

        por_muestra = (sc_te.groupBy("sample_id", "label")
                       .agg(F.max("alarma").alias("detectada"),
                            F.min(F.when(F.col("alarma") == 1, F.col("t"))).alias("retardo"),
                            F.avg("excede").alias("frac_excede"))
                       .withColumn("fold", F.lit(fold)))
        res_pca.append(por_muestra.toPandas())

        # --- Isolation Forest sobre estadísticos por ventana (agregación en Spark) ---
        aggs = [F.mean("z" + str(i)).alias("m" + str(i)) for i in range(p)] + \
               [F.stddev("z" + str(i)).alias("s" + str(i)) for i in range(p)]
        def ventana(d):
            d = d.select("sample_id", "label", *[F.col("z")[i].alias("z" + str(i)) for i in range(p)])
            return d.groupBy("sample_id", "label").agg(*aggs).toPandas()
        wtr, wte = ventana(tr_z), ventana(te_z)
        fcols = [c for c in wtr.columns if c[0] in "ms" and c[1:].isdigit()]
        esc = StandardScaler().fit(wtr[fcols])
        iso = IsolationForest(n_estimators=300, random_state=42).fit(esc.transform(wtr[fcols]))
        s_tr = -iso.score_samples(esc.transform(wtr[fcols]))
        thr_if = np.quantile(s_tr, 0.95)
        s_te = -iso.score_samples(esc.transform(wte[fcols]))
        out = wte[["sample_id", "label"]].copy()
        out["score_if"], out["detectada_if"], out["fold"] = s_te, (s_te > thr_if).astype(int), fold
        res_if.append(out)

        np.savez(GOLD / f"modelo_fold{fold}.npz", cols=np.array(utiles),
                 mu=np.array([mu[c] for c in utiles]), sd=np.array([sd[c] for c in utiles]),
                 P=P, lam=lam, thr_t2=thr_t2, thr_spe=thr_spe)
        info_folds.append({"fold": fold, "componentes": k, "varianza_retenida": float(np.cumsum(ev)[k - 1]),
                           "umbral_t2": float(thr_t2), "umbral_spe": float(thr_spe),
                           "umbral_if": float(thr_if)})
        print(f"fold {fold}: k={k}, thr T2={thr_t2:.2f}, SPE={thr_spe:.2f}  ({time.time()-t0:.0f}s)", flush=True)

    pca_df = pd.concat(res_pca).sort_values("sample_id")
    if_df = pd.concat(res_if).sort_values("sample_id")
    GOLD.mkdir(parents=True, exist_ok=True)
    pca_df.to_parquet(GOLD / "deteccion_pca_por_muestra.parquet")
    if_df.to_parquet(GOLD / "deteccion_if_por_muestra.parquet")
    pd.DataFrame(info_folds).to_csv(SALIDAS / "tablas" / "parametros_por_fold.csv", index=False)
    print(f"Gold completo en {time.time()-t0:.0f}s")
    spark.stop()

if __name__ == "__main__":
    main()
