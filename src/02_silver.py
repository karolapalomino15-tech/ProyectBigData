"""Bronze -> Silver (Spark).
- Calidad: conteo de nulos/infinitos y duplicados (sample_id, t).
- Se eliminan variables constantes (desviación estándar = 0).
- Se agrega es_falla y se guarda particionado por fold."""
import json
from pyspark.sql import functions as F
from config import get_spark, BRONZE, SILVER, SALIDAS, COLS

def main():
    spark = get_spark("TEP-silver")
    df = spark.read.parquet(str(BRONZE))
    total = df.count()

    malos = df.select([F.sum(F.when(F.col(c).isNull() | F.isnan(c) |
                       F.col(c).isin(float("inf"), float("-inf")), 1).otherwise(0)).alias(c)
                       for c in COLS]).first().asDict()
    dup = total - df.select("sample_id", "t").distinct().count()
    sds = df.select([F.stddev(c).alias(c) for c in COLS]).first().asDict()
    constantes = [c for c in COLS if sds[c] is None or sds[c] < 1e-9]
    utiles = [c for c in COLS if c not in constantes]

    silver = (df.drop(*constantes)
                .withColumn("es_falla", (F.col("label") > 0).cast("int")))
    silver.write.mode("overwrite").partitionBy("fold").parquet(str(SILVER))

    resumen = {"filas": total, "valores_invalidos": int(sum(malos.values())),
               "duplicados": int(dup), "variables_constantes": constantes,
               "variables_utiles": utiles, "n_utiles": len(utiles)}
    (SALIDAS / "tablas").mkdir(parents=True, exist_ok=True)
    (SALIDAS / "tablas" / "calidad_silver.json").write_text(json.dumps(resumen, indent=2))
    print(json.dumps(resumen, indent=2))
    spark.stop()

if __name__ == "__main__":
    main()
