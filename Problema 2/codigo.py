import io
import zipfile
import numpy as np
import pandas as pd
import requests
import xgboost as xgb
from sklearn.linear_model import QuantileRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

# ==========================================
# 0. DESCARGA Y LECTURA DE DATOS
# ==========================================
url = "https://archive.ics.uci.edu/static/public/560/seoul+bike+sharing+demand.zip"
print("Descargando dataset...")
response = requests.get(url)
z = zipfile.ZipFile(io.BytesIO(response.content))

# El archivo CSV dentro del zip suele tener codificación ISO-8859-1 o cp1252 debido al idioma coreano/caracteres especiales
csv_filename = [f for f in z.namelist() if f.endswith('.csv')][0]
with z.open(csv_filename) as f:
    df = pd.read_csv(f, encoding='ISO-8859-1')

print("Datos cargados correctamente. Dimensiones:", df.shape)

# ==========================================
# a) PREPROCESAMIENTO Y PARTICIÓN
# ==========================================
# Limpieza de nombres de columnas por si acaso
df.columns = df.columns.str.strip()

# Convertir la columna de fecha
df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')


print(f"Filas totales antes de filtrar Functioning Day == 'No': {len(df)}")
df = df[df['Functioning Day'] == 'Yes'].copy()

# Extracción de atributos temporales útiles
df['Hour'] = df['Date'].dt.hour 

# Transformar variables categóricas
df['Seasons'] = df['Seasons'].astype('category')
df['Holiday'] = df['Holiday'].astype('category')


test_mask = (df['Date'].dt.year == 2018) & (df['Date'].dt.month.isin([10, 11]))

train_df = df[~test_mask].copy()
test_df = df[test_mask].copy()

print(f"Tamaño entrenamiento: {len(train_df)} | Tamaño prueba: {len(test_df)}")


features = [
    'Hour', 'Temperature(°C)', 'Humidity(%)', 'Wind speed (m/s)',
    'Visibility (10m)', 'Dew point temperature(°C)', 'Solar Radiation (MJ/m2)',
    'Rainfall(mm)', 'Snowfall (cm)', 'Seasons', 'Holiday'
]
target = 'Rented Bike Count'

X_train = train_df[features]
y_train = train_df[target].to_numpy(dtype=float)

X_test = test_df[features]
y_test = test_df[target].to_numpy(dtype=float)



# Parámetros para la pérdida asimétrica (Cuantiles, ej: tau = 0.75 o el que se requiera)
tau = 0.8
alpha = 4    
beta = 1.0   


# ==========================================
# b) MODELOS
# ==========================================

# M1: XGBoost con pérdida cuadrática (reg:squarederror)
print("\nEntrenando M1 (MSE)...")
m1 = xgb.XGBRegressor(objective='reg:squarederror', enable_categorical=True, n_estimators=100, random_state=218207462)
m1.fit(X_train, y_train)


# M2: XGBoost con la pérdida analizada en el inciso (c) (Segunda derivada cero / constante)
def custom_objective_c(preds, dtrain):
    labels = dtrain.get_label()
    residuals = labels - preds
    
    grad = np.where(residuals > 0, -alpha, beta)
   
    hess = np.ones_like(preds) * 1e-5 # 
    return grad, hess

print("Entrenando M2 (Objetivo C - h_i aprox 0)...")
dtrain_m2 = xgb.DMatrix(X_train, label=y_train, enable_categorical=True)
dtest_m2 = xgb.DMatrix(X_test, label=y_test, enable_categorical=True)

params_m2 = {'max_depth': 4, 'eta': 0.1, 'min_child_weight': 1}
# Nota: Con h=0 y min_child_weight=1, los árboles no crecen bien como predice la teoría.
bst_m2 = xgb.train(params_m2, dtrain_m2, num_boost_round=50, obj=custom_objective_c)


# M3: XGBoost con tu arreglo del inciso (e) (Función objetivo suavizada con sigmoide)
# g_i = beta - (alpha + beta) * sigma(z_i)
# h_i = ((alpha + beta) / s) * sigma(z_i) * (1 - sigma(z_i))
def custom_objective_e(preds, dtrain, s=10.0):
    labels = dtrain.get_label()
    z = (labels - preds) / s
    # Sigmoide estable
    sig = 1.0 / (1.0 + np.exp(-np.clip(z, -50, 50)))
    
    grad = beta - (alpha + beta) * sig
    hess = ((alpha + beta) / s) * sig * (1.0 - sig)
    # Evitar hessianos nulos añadiendo un piso mínimo de estabilidad
    hess = np.maximum(hess, 1e-4)
    return grad, hess

print("Entrenando M3 (Arreglo E - Suavizado con sigmoide)...")
params_m3 = {'max_depth': 6, 'eta': 0.1, 'min_child_weight': 10, 'lambda': 1.0}
bst_m3 = xgb.train(params_m3, dtrain_m2, num_boost_round=100, obj=lambda p, d: custom_objective_e(p, d, s=20.0))


# M4: XGBoost con objetivo de cuantiles integrado (reg:quantileerror)
print("Entrenando M4 (XGBoost Quantile nativo)...")
m4 = xgb.XGBRegressor(
    objective='reg:quantileerror',
    quantile_alpha=tau,
    enable_categorical=True,
    n_estimators=100,
    random_state=218207462
)
m4.fit(X_train, y_train)


# M5: Regresión lineal de cuantiles (QuantileRegressor de scikit-learn)

print("Entrenando M5 (QuantileRegressor sklearn)...")
X_train_encoded = pd.get_dummies(X_train, drop_first=True)
X_test_encoded = pd.get_dummies(X_test, drop_first=True)
# Alinear columnas por si difieren
X_test_encoded = X_test_encoded.reindex(columns=X_train_encoded.columns, fill_value=0)

m5 = QuantileRegressor(quantile=tau, alpha=0.01, solver= 'highs')
m5.fit(X_train_encoded, y_train)


# ==========================================
# c) EVALUACIÓN
# ==========================================
# Obtener predicciones
preds_m1 = m1.predict(X_test)
preds_m2 = bst_m2.predict(dtest_m2)
preds_m3 = bst_m3.predict(dtest_m2)
preds_m4 = m4.predict(X_test)
preds_m5 = m5.predict(X_test_encoded)

models_dict = {
    "M1 (MSE)": preds_m1,
    "M2 (Teoría C - h=0)": preds_m2,
    "M3 (Tu Arreglo E)": preds_m3,
    "M4 (XGB Quantile)": preds_m4,
    "M5 (SKL QuantileReg)": preds_m5
}

evaluation_results = []

for name, preds in models_dict.items():
    # Costo promedio por hora con la pérdida asimétrica real L (Pinball loss / Costo de cuantil)
    errors = y_test - preds
    pinball_loss = np.mean(np.maximum(alpha * errors, (alpha - 1) * errors))
    
   
 
    coverage = np.mean(y_test <= preds)
    
    evaluation_results.append({
        "Modelo": name,
        "Costo Promedio (Pinball Loss)": pinball_loss,
        "Cobertura Real": coverage,
        "Cobertura Objetivo (tau)": tau
    })

eval_df = pd.DataFrame(evaluation_results)
print("\n--- RESULTADOS DE EVALUACIÓN ---")
print(eval_df.to_string(index=False))

# Gráfica de cobertura por hora del día para M3  y M4
test_df_eval = test_df.copy()
test_df_eval['Pred_M3'] = preds_m3
test_df_eval['Pred_M4'] = preds_m4
test_df_eval['Under_M3'] = (test_df_eval[target] <= test_df_eval['Pred_M3']).astype(int)


hourly_coverage = test_df_eval.groupby('Hour')['Under_M3'].mean().reindex(range(24), fill_value=0)

plt.figure(figsize=(10, 5))
plt.plot(hourly_coverage.index, hourly_coverage.to_numpy(dtype=float) , marker='o', label='Cobertura M3 (Arreglo E)')
plt.axhline(y=tau, color='r', linestyle='--', label=f'Cobertura deseada (tau={tau})')
plt.title('Cobertura por hora del día')
plt.xlabel('Hora del día (0-23)')
plt.ylabel('Fracción con y_real <= y_pred')
plt.legend()
plt.grid(True)
plt.savefig('cobertura_por_hora.png')
print("\nGráfica de cobertura guardada como 'cobertura_por_hora.png'.")
