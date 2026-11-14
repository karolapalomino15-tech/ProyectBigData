"""Configuración central del pipeline TEP (arquitectura Medallón)."""
from pathlib import Path
from pyspark.sql import SparkSession

RAIZ = Path(__file__).resolve().parents[1]
LANDING = RAIZ / "data" / "landing" / "TEPDataset_Mode1.pickle"
BRONZE = RAIZ / "data" / "bronze"
SILVER = RAIZ / "data" / "silver"
GOLD = RAIZ / "data" / "gold"
SALIDAS = RAIZ / "outputs"

N_VARS = 34                      # variables en el pickle
COLS = [f"v{i:02d}" for i in range(N_VARS)]

# Parámetros de detección
PERCENTIL_UMBRAL = 0.995         # umbral T2/SPE sobre datos normales de entrenamiento
K_CONSECUTIVAS = 5               # alarma si K pasos seguidos superan el umbral
VARIANZA_PCA = 0.90              # varianza a retener en PCA


def get_spark(nombre="TEP-BigData"):
    return (SparkSession.builder.appName(nombre)
            .master("local[*]")
            .config("spark.driver.memory", "2g")
            .config("spark.sql.shuffle.partitions", "8")
            .config("spark.sql.execution.arrow.pyspark.enabled", "true")
            .config("spark.ui.showConsoleProgress", "false")
            .getOrCreate())
