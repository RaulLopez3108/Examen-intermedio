import os
import sys

# Suprimir advertencias y configurar entorno para evitar conflictos con NumPy en Windows
import warnings

warnings.filterwarnings("ignore")

import urllib.request
import zipfile
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def descargar_y_limpiar_vinos():
  try:
    print("Iniciando descarga y preprocesamiento...")
    os.makedirs("dataset/wine_data", exist_ok=True)

    url = "https://archive.ics.uci.edu/static/public/109/wine.zip"
    zip_path = "dataset/wine.zip"
    extract_path = "dataset/wine_data"

    if not os.path.exists(zip_path):
      print("Descargando dataset de Vinos de Italia...")
      urllib.request.urlretrieve(url, zip_path)

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
      zip_ref.extractall(extract_path)
      print("Dataset descomprimido correctamente.")

    data_file = os.path.join(extract_path, "wine.data")
    if not os.path.exists(data_file):
      for root, dirs, files in os.walk(extract_path):
        if "wine.data" in files:
          data_file = os.path.join(root, "wine.data")

    columnas = [
        "Clase",
        "Alcohol",
        "Malic_Acid",
        "Ash",
        "Alcalinity_of_ash",
        "Magnesium",
        "Total_phenols",
        "Flavanoids",
        "Nonflavanoid_phenols",
        "Proanthocyanins",
        "Color_intensity",
        "Hue",
        "OD280_OD315_of_diluted_wines",
        "Proline",
    ]

    df = pd.read_csv(data_file, header=None, names=columnas)

    if df.isnull().sum().sum() > 0:
      df.fillna(df.median(), inplace=True)

    y = df["Clase"].values
    X = df.drop(columns=["Clase"]).values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y

  except Exception as e:
    print(f"Ocurrió un error durante la ejecución: {e}")
    sys.exit(1)


if __name__ == "__main__":
  X, y = descargar_y_limpiar_vinos()

  print("\n--- ¡DATOS CARGADOS Y LIMPIOS CON ÉXITO! ---")
  print("Forma de la matriz X (características químicas):", X.shape)
  print("Forma del vector y (clases de los vinos):", y.shape)
  print("\nPrimeras 2 filas de características normalizadas (X):")
  print(X[:2])