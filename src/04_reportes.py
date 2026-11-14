"""Gold -> tablas y figuras para el informe."""
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, balanced_accuracy_score
from config import GOLD, SALIDAS

T, F = SALIDAS / "tablas", SALIDAS / "figuras"
T.mkdir(parents=True, exist_ok=True); F.mkdir(parents=True, exist_ok=True)
p = pd.read_parquet(GOLD / "deteccion_pca_por_muestra.parquet")
i = pd.read_parquet(GOLD / "deteccion_if_por_muestra.parquet")
m = p.merge(i[["sample_id", "score_if", "detectada_if"]], on="sample_id")
m["real"] = (m.label > 0).astype(int)

# ---- métricas globales (nivel ventana) ----
filas = []
for nombre, col in [("PCA (T2/SPE, K=5 pasos)", "detectada"), ("Isolation Forest (ventana)", "detectada_if")]:
    y, yh = m.real, m[col].astype(int)
    filas.append({"modelo": nombre,
                  "recall (fallas detectadas)": recall_score(y, yh),
                  "falsas alarmas (normales con alarma)": yh[y == 0].mean(),
                  "precision": precision_score(y, yh), "F1": f1_score(y, yh),
                  "exactitud balanceada": balanced_accuracy_score(y, yh)})
glob = pd.DataFrame(filas).round(3)
glob.to_csv(T / "metricas_globales.csv", index=False)

# ---- por tipo de falla ----
pf = (m[m.label > 0].groupby("label")
      .agg(deteccion_pca=("detectada", "mean"), deteccion_if=("detectada_if", "mean"),
           retardo_mediano_pca=("retardo", "median"), retardo_p90_pca=("retardo", lambda s: s.quantile(.9)))
      .round(2).reset_index().rename(columns={"label": "falla"}))
pf.to_csv(T / "metricas_por_falla.csv", index=False)

# ---- por fold (estabilidad de la validación cruzada) ----
pfold = (m.assign(fa_pca=lambda d: d.detectada.where(d.real == 0), fa_if=lambda d: d.detectada_if.where(d.real == 0),
                  dr_pca=lambda d: d.detectada.where(d.real == 1), dr_if=lambda d: d.detectada_if.where(d.real == 1))
         .groupby("fold")[["dr_pca", "fa_pca", "dr_if", "fa_if"]].mean().round(3).reset_index())
pfold.to_csv(T / "metricas_por_fold.csv", index=False)

# ---- Figura 1: detección por falla ----
x = np.arange(len(pf)); w = 0.4
fig, ax = plt.subplots(figsize=(11, 4))
ax.bar(x - w/2, pf.deteccion_pca, w, label="PCA (T²/SPE)")
ax.bar(x + w/2, pf.deteccion_if, w, label="Isolation Forest")
ax.set_xticks(x); ax.set_xticklabels(pf.falla); ax.set_xlabel("Tipo de falla"); ax.set_ylabel("Fracción detectada")
ax.set_title("Tasa de detección por tipo de falla (validación cruzada, 5 folds)"); ax.legend()
fig.tight_layout(); fig.savefig(F / "deteccion_por_falla.png", dpi=150); plt.close(fig)

# ---- Figura 2: retardo de detección ----
d = pf.dropna(subset=["retardo_mediano_pca"])
fig, ax = plt.subplots(figsize=(11, 4))
ax.bar(d.falla.astype(str), d.retardo_mediano_pca)
ax.set_xlabel("Tipo de falla"); ax.set_ylabel("Pasos hasta la alarma (mediana)")
ax.set_title("Retardo de detección del modelo PCA en fallas detectadas")
fig.tight_layout(); fig.savefig(F / "retardo_deteccion.png", dpi=150); plt.close(fig)

# ---- Figura 3: trazas T2 / SPE en ejemplos (modelo guardado del fold 1) ----
import pyarrow.dataset as ds
from config import SILVER
mod = np.load(GOLD / "modelo_fold1.npz", allow_pickle=True)
cols = list(mod["cols"])
tab = ds.dataset(str(SILVER), format="parquet", partitioning="hive").to_table(
    filter=(ds.field("fold") == 1) & ds.field("label").isin([0, 1, 6, 15]),
    columns=["sample_id", "label", "t"] + cols).to_pandas()
fig, axs = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
for lab, color in [(0, "tab:green"), (1, "tab:red"), (6, "tab:orange"), (15, "tab:purple")]:
    sid = tab[tab.label == lab].sample_id.iloc[0]
    s = tab[tab.sample_id == sid].sort_values("t")
    X = (s[cols].to_numpy() - mod["mu"]) / mod["sd"]
    Tm = X @ mod["P"]
    t2 = (Tm ** 2 / mod["lam"]).sum(1); spe = (X ** 2).sum(1) - (Tm ** 2).sum(1)
    nombre = "Normal" if lab == 0 else f"Falla {lab}"
    axs[0].plot(s.t, t2, color=color, label=nombre, lw=1); axs[1].plot(s.t, spe, color=color, lw=1)
axs[0].axhline(float(mod["thr_t2"]), color="k", ls="--", lw=1); axs[1].axhline(float(mod["thr_spe"]), color="k", ls="--", lw=1)
axs[0].set_yscale("log"); axs[1].set_yscale("log")
axs[0].set_ylabel("T²"); axs[1].set_ylabel("SPE"); axs[1].set_xlabel("Paso de tiempo")
axs[0].legend(ncol=4); axs[0].set_title("Estadísticos T² y SPE por paso (línea punteada = umbral, fold 1)")
fig.tight_layout(); fig.savefig(F / "trazas_t2_spe.png", dpi=150); plt.close(fig)

print(glob.to_string(index=False)); print(); print(pfold.to_string(index=False))
