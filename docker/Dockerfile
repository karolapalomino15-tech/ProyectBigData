FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1
# Spark necesita Java; default-jre-headless instala la versión LTS disponible en la imagen base (17 o 21)
RUN apt-get update && apt-get install -y --no-install-recommends default-jre-headless procps \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY src/ src/
COPY run_all.sh .
# El pickle se monta como volumen: ./data/landing/TEPDataset_Mode1.pickle
CMD ["bash", "run_all.sh"]
