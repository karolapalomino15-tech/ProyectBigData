"""Genera muestras pequeñas (20 a 100 filas) de cada etapa para el repositorio (carpeta muestras/).
Requiere haber ejecutado antes los scripts 01 a 05. Las muestras usan las mismas simulaciones en cada etapa
para poder seguir un registro desde el dato crudo hasta el resultado."""
import pickle
import numpy as np, pandas as pd
import pyarrow.dataset as ds
from config import RAIZ, LANDING, BRONZE, SILVER, GOLD, COLS

M = RAIZ / "muestras"
CLASES = [0, 1, 4, 6, 15]      # normal y cuatro fallas (dos claras, una media y una indetectable)
PASOS = 10                      # primeros pasos de cada simulación
def guardar(df, ruta):
    ruta.parent.mkdir(parents=True, exist_ok=True)
    df.round(6).to_csv(ruta, index=False); print(f"{ruta.relative_to(RAIZ)}: {len(df)} filas")

# 1) CRUDO: directamente del pickle
d = pickle.load(open(LANDING, "rb"))
S, L, F = d["Signals"], d["Labels"], d["Folds"]
fold_de = np.zeros(len(L), dtype=int)
for k, idx in F.items(): fold_de[idx] = int(k.split()[-1])
ids = [int(np.where(L == c)[0][0]) for c in CLASES]
crudo = pd.concat([pd.DataFrame(S[i, :PASOS, :], columns=COLS).assign(sample_id=i, label=int(L[i]), fold=int(fold_de[i]), t=np.arange(PASOS)) for i in ids])
crudo = crudo[["sample_id", "label", "fold", "t"] + COLS].sort_values(["sample_id", "t"])
guardar(crudo, M / "00_crudo" / "muestra_cruda.csv")

# 2) BRONZE
b = ds.dataset(str(BRONZE), format="parquet").to_table(filter=ds.field("sample_id").isin(ids) & (ds.field("t") < PASOS)).to_pandas().sort_values(["sample_id", "t"])
guardar(b, M / "01_bronze" / "muestra_bronze.csv")

# 3) SILVER
s = ds.dataset(str(SILVER), format="parquet", partitioning="hive").to_table(filter=ds.field("sample_id").isin(ids) & (ds.field("t") < PASOS)).to_pandas().sort_values(["sample_id", "t"])
guardar(s, M / "02_silver" / "muestra_silver.csv")

# 4) GOLD: resultados por simulación (2 por clase) y T2/SPE por paso
p = pd.read_parquet(GOLD / "deteccion_pca_por_muestra.parquet"); i = pd.read_parquet(GOLD / "deteccion_if_por_muestra.parquet")
g = p.merge(i[["sample_id", "score_if", "detectada_if"]], on="sample_id").sort_values(["label", "sample_id"]).groupby("label").head(2)
guardar(g.rename(columns={"detectada": "detectada_pca", "retardo": "retardo_pca"}), M / "03_gold" / "muestra_gold_deteccion.csv")
filas = []
for sid in ids:
    mod = np.load(GOLD / f"modelo_fold{int(fold_de[sid])}.npz", allow_pickle=True)
    cols = list(mod["cols"]); sub = s[s.sample_id == sid].sort_values("t")
    X = (sub[cols].to_numpy() - mod["mu"]) / mod["sd"]; T = X @ mod["P"]
    t2 = (T ** 2 / mod["lam"]).sum(1); spe = (X ** 2).sum(1) - (T ** 2).sum(1)
    filas.append(pd.DataFrame({"sample_id": sid, "label": int(L[sid]), "t": sub.t.values, "t2": t2, "spe": spe,
                               "excede": ((t2 > float(mod["thr_t2"])) | (spe > float(mod["thr_spe"]))).astype(int)}))
guardar(pd.concat(filas), M / "03_gold" / "muestra_gold_t2_spe_por_paso.csv")

# 5) STREAMING: primer micro-lote emitido y bitácora
mb = pd.read_parquet(RAIZ / "data" / "stream_in" / "lote_0000.parquet").sort_values(["sample_id", "t"]).groupby("sample_id").head(15)
guardar(mb, M / "04_streaming" / "muestra_microlote.csv")
