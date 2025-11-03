# ==========================================================
# CLASIFICACIÓN - REGRESIÓN LOGÍSTICA
# Dataset: consumo_agua_comunas_70.csv
# ==========================================================
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from limpiador_datos import limpiador_csv
import matplotlib.pyplot as plt
import seaborn as sns

# Ruta del archivo
ruta = 'consumo_agua_comunas.csv'
separador = ','

# Funcion de limpieza
exito, df_consumo_agua, mensaje = limpiador_csv(ruta, sep=separador)

if exito:
    print("La limpieza se realizo con exito.")
    print(f"Mensaje:{mensaje}")
else:
    print(f"La limpieza fallo: {mensaje}")

#1. Crear variable binaria [Consumo_alto]: 1 y 0
promedio_consumo = df_consumo_agua["consumo_m3"].mean()
df_consumo_agua["consumo_alto"] = (df_consumo_agua["consumo_m3"] >= promedio_consumo).astype(int) 

print(f"Promedio de consumo: {promedio_consumo:.2f} m³")
print(df_consumo_agua["consumo_alto"].value_counts(), "\n")

#3. Datos predictorios
X = df_consumo_agua[["poblacion", "ingreso_promedio", "temperatura_promedio", "precipitacion_mm", "mes"]]
y = df_consumo_agua["consumo_alto"]

#4. Creacion de variables de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#5. Creacion y entrenamiento del modelo
modelo = LogisticRegression(max_iter=500)
modelo.fit(X_train, y_train)

#6. Predicciones del modelo
y_pred = modelo.predict(X_test)

#7. Evaluación de acertividad del modelo
acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print(f"Exactitud del modelo: {acc:.2f}")
print("\nMatriz de confusión:\n", cm)
print("\nReporte de clasificación:\n", classification_report(y_test, y_pred))

#8. Creacion de matriz de confusión
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicción")
plt.ylabel("Real")
plt.title("Matriz de Confusión - Regresión Logística")
plt.show()
