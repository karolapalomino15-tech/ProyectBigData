"""Landing -> Bronze.
El pickle (arreglo 3D) no es legible por Spark; se aplana a formato tabular
(una fila = una muestra en un paso de tiempo) y se guarda como Parquet por lotes.
Bronze = dato crudo, sin ninguna transformación de valores."""
import pickle
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from config import LANDING, BRONZE, COLS

LOTE = 100  # simulaciones por archivo parquet

def main():
    with open(LANDING, "rb") as f:
        d = pickle.load(f)
    S, L, folds = d["Signals"], d["Labels"], d["Folds"]
    n, T, v = S.shape
    fold_de = np.zeros(n, dtype=np.int64)
    for k, idx in folds.items():
        fold_de[idx] = int(k.split()[-1])

    BRONZE.mkdir(parents=True, exist_ok=True)
    for f in BRONZE.glob("*.parquet"):
        f.unlink()
    for j, ini in enumerate(range(0, n, LOTE)):
        fin = min(ini + LOTE, n)
        m = fin - ini
        cols = {
            "sample_id": np.repeat(np.arange(ini, fin), T),
            "label": np.repeat(L[ini:fin], T),
            "fold": np.repeat(fold_de[ini:fin], T),
            "t": np.tile(np.arange(T), m),
        }
        X = S[ini:fin].reshape(-1, v)
        for i, c in enumerate(COLS):
            cols[c] = X[:, i]
        pq.write_table(pa.table(cols), BRONZE / f"part-{j:04d}.parquet", compression="snappy")
    print(f"Bronze listo: {n*T:,} filas, {v} variables, {n} simulaciones")

if __name__ == "__main__":
    main()
