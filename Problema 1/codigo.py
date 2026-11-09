import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from cargar_dataset import descargar_y_limpiar_vinos

print("=== INICIO DE LA RESOLUCIÓN COMPLETA (INCISOS A, B y C) ===")


# ==========================================
# INCISO (A): Gap Statistic y Simulaciones
# ==========================================
print("\n--- EJECUTANDO INCISO (A): Gap Statistic ---")

a = 2
b = 6  # Parámetro personal
d = 2 + (b % 3)  
n = 200 + 50 * b  
k_max_a = 2
B_a = 50


def criterio_gap_statistic(X, k_max=2, B=50):
  n_samples, d_dim = X.shape
  W_k = np.zeros(k_max + 1)
  for k in range(1, k_max + 1):
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=218207462).fit(X)
    labels = kmeans.labels_
    centers = kmeans.cluster_centers_
    w = 0.0
    for j in range(k):
      cp = X[labels == j]
      if len(cp) > 0:
        w += np.sum((cp - centers[j]) ** 2)
    W_k[k] = w

  U, S, Vt = np.linalg.svd(X, full_matrices=False)
  X_prime = X @ Vt.T
  a_j = X_prime.min(axis=0)
  b_j = X_prime.max(axis=0)

  log_W_ref = np.zeros((k_max + 1, B))
  for b_idx in range(B):
    Z = np.zeros((n_samples, d_dim))
    for j in range(d_dim):
      Z[:, j] = np.random.uniform(a_j[j], b_j[j], size=n_samples)
    X_ref = Z @ Vt

    for k in range(1, k_max + 1):
      kmeans_ref = KMeans(n_clusters=k, n_init=10, random_state=218207462).fit(X_ref)
      labels_ref = kmeans_ref.labels_
      centers_ref = kmeans_ref.cluster_centers_
      w_ref = 0.0
      for j in range(k):
        cp = X_ref[labels_ref == j]
        if len(cp) > 0:
          w_ref += np.sum((cp - centers_ref[j]) ** 2)
      log_W_ref[k, b_idx] = np.log(w_ref) if w_ref > 0 else 0.0

  Gap = np.zeros(k_max + 1)
  s_k = np.zeros(k_max + 1)
  L_bar = np.zeros(k_max + 1)

  for k in range(1, k_max + 1):
    L_bar[k] = np.mean(log_W_ref[k, :])
    Gap[k] = L_bar[k] - np.log(W_k[k])
    sd_k = np.sqrt(
        np.sum((log_W_ref[k, :] - L_bar[k]) ** 2) / (B - 1) if B > 1 else 0.0
    )
    s_k[k] = sd_k * np.sqrt(1.0 + 1.0 / B)

  k_star = k_max
  for k in range(1, k_max):
    if Gap[k] >= Gap[k + 1] - s_k[k + 1]:
      k_star = k
      break

  return k_star > 1


num_simulaciones = 100  

# Simulación de Falsas Alarmas (Estándar)
fa_estandar = 0
for _ in range(num_simulaciones):
  X_norm = np.random.normal(loc=0.0, scale=1.0, size=(n, d))
  if criterio_gap_statistic(X_norm, k_max_a, B_a):
    fa_estandar += 1
print(f" -> Tasa Falsas Alarmas (Normal Estándar): {(fa_estandar/num_simulaciones)*100}%")

# Curva de Detección frente a mu
mu_valores = np.linspace(0.0, 4.0, 8)
tasas_deteccion = []
for mu in mu_valores:
  det_count = 0
  for _ in range(num_simulaciones):
    n1, n2 = n // 2, n - (n // 2)
    mean1, mean2 = np.zeros(d), np.zeros(d)
    mean2[0] += mu
    X1 = np.random.normal(loc=mean1, scale=1.0, size=(n1, d))
    X2 = np.random.normal(loc=mean2, scale=1.0, size=(n2, d))
    X_mix = np.vstack([X1, X2])
    if criterio_gap_statistic(X_mix, k_max_a, B_a):
      det_count += 1
  tasas_deteccion.append((det_count / num_simulaciones) * 100)

plt.figure(figsize=(7, 4))
plt.plot(
    mu_valores,
    tasas_deteccion,
    marker="o",
    linestyle="-",
    color="navy",
    linewidth=2,
)
plt.title(
    f"Curva de Detección (Gap Statistic) vs $\\mu$ (d={d}, n={n})", fontsize=11
)
plt.xlabel("Separación $\\mu$")
plt.ylabel("Porcentaje de Detección (%)")
plt.grid(True, linestyle="--", alpha=0.7)
plt.savefig("curva_deteccion_gap_alg1.png", dpi=300, bbox_inches="tight")
plt.close()
print(" -> Gráfica del Inciso (A) guardada.")


# ==========================================
# INCISO (B): La trampa visual con t-SNE
# ==========================================
print("\n--- EJECUTANDO INCISO (B): La trampa visual (t-SNE) ---")

d_b = 10
n_b = 300
np.random.seed(42)
X_normal_10d = np.random.normal(loc=0.0, scale=1.0, size=(n_b, d_b))
perplejidades = [5, 30, 50]

plt.figure(figsize=(14, 4))
for i, perp in enumerate(perplejidades):
  tsne = TSNE(n_components=2, perplexity=perp, random_state=218207462)
  X_embedded = tsne.fit_transform(X_normal_10d)

  plt.subplot(1, len(perplejidades), i + 1)
  plt.scatter(
      X_embedded[:, 0],
      X_embedded[:, 1],
      color="purple",
      alpha=0.7,
      edgecolors="k",
  )
  plt.title(f"t-SNE (Perplejidad = {perp})", fontsize=11)
  plt.grid(True, linestyle="--", alpha=0.5)

plt.suptitle(
    "La trampa visual: t-SNE en una sola normal en $\\mathbb{R}^{10}$",
    fontsize=13,
    weight="bold",
)
plt.tight_layout()
plt.savefig("trampa_visual_tsne.png", dpi=300, bbox_inches="tight")
plt.close()
print(" -> Gráfica del Inciso (B) guardada.")


# ==========================================
# INCISO (C): Datos Reales (Vinos de Italia)
# ==========================================
print("\n--- EJECUTANDO INCISO (C): Datos Reales de Vinos ---")


X_scaled, y_real = descargar_y_limpiar_vinos()

data_file = "dataset/wine_data/wine.data"
if not os.path.exists(data_file):
  for root, dirs, files in os.walk("dataset/wine_data"):
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
df_original = pd.read_csv(data_file, header=None, names=columnas)
X_raw = df_original.drop(columns=["Clase"])


k_max_c = 5
B_c = 50
n_samples_c, d_dim_c = X_scaled.shape

W_k_c = np.zeros(k_max_c + 1)
for k in range(1, k_max_c + 1):
  kmeans = KMeans(n_clusters=k, n_init=10, random_state=218207462).fit(X_scaled)
  labels = kmeans.labels_
  centers = kmeans.cluster_centers_
  w = 0.0
  for j in range(k):
    cp = X_scaled[labels == j]
    if len(cp) > 0:
      w += np.sum((cp - centers[j]) ** 2)
  W_k_c[k] = w

U_c, S_c, Vt_c = np.linalg.svd(X_scaled, full_matrices=False)
X_prime_c = X_scaled @ Vt_c.T
a_j_c = X_prime_c.min(axis=0)
b_j_c = X_prime_c.max(axis=0)

log_W_ref_c = np.zeros((k_max_c + 1, B_c))
for b_idx in range(B_c):
  Z = np.zeros((n_samples_c, d_dim_c))
  for j in range(d_dim_c):
    Z[:, j] = np.random.uniform(a_j_c[j], b_j_c[j], size=n_samples_c)
  X_ref = Z @ Vt_c

  for k in range(1, k_max_c + 1):
    kmeans_ref = KMeans(n_clusters=k, n_init=10, random_state=218207462).fit(X_ref)
    labels_ref = kmeans_ref.labels_
    centers_ref = kmeans_ref.cluster_centers_
    w_ref = 0.0
    for j in range(k):
      cp = X_ref[labels_ref == j]
      if len(cp) > 0:
        w_ref += np.sum((cp - centers_ref[j]) ** 2)
    log_W_ref_c[k, b_idx] = np.log(w_ref) if w_ref > 0 else 0.0

Gap_c = np.zeros(k_max_c + 1)
s_k_c = np.zeros(k_max_c + 1)
L_bar_c = np.zeros(k_max_c + 1)

for k in range(1, k_max_c + 1):
  L_bar_c[k] = np.mean(log_W_ref_c[k, :])
  Gap_c[k] = L_bar_c[k] - np.log(W_k_c[k])
  sd_k = np.sqrt(
      np.sum((log_W_ref_c[k, :] - L_bar_c[k]) ** 2) / (B_c - 1)
      if B_c > 1
      else 0.0
  )
  s_k_c[k] = sd_k * np.sqrt(1.0 + 1.0 / B_c)

k_star_c = k_max_c
for k in range(1, k_max_c):
  if Gap_c[k] >= Gap_c[k + 1] - s_k_c[k + 1]:
    k_star_c = k
    break

print(f"\n >>> ¡NÚMERO ÓPTIMO DE GRUPOS DETECTADO (k*): {k_star_c} <<< \n")


kmeans_final = KMeans(n_clusters=k_star_c, n_init=20, random_state=218207462).fit(
    X_scaled
)
X_raw["Grupo_Asignado"] = kmeans_final.labels_ + 1

print("--- DESCRIPCIÓN DE LOS GRUPOS (PROMEDIOS EN VARIABLES ORIGINALES) ---")
print(X_raw.groupby("Grupo_Asignado").mean().T.to_string())

X_raw["Clase_Real"] = y_real
print("\n--- MATRIZ DE CONTINGENCIA (Grupos Detectados vs Clases Reales) ---")
print(
    pd.crosstab(
        X_raw["Grupo_Asignado"],
        X_raw["Clase_Real"],
        rownames=["Grupo Detectado"],
        colnames=["Clase Real UCI"],
    )
)

print("\n=== ¡TODOS LOS INCISOS COMPLETADOS CON ÉXITO! ===")
