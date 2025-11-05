import pandas as pd
from limpiador_datos import limpiador_csv
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt

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


#1. Vista inicial del DataFrame
print("Vista general del dataset:\n", df_consumo_agua.head(), "\n")
print("Años considerados para el estudio:", df_consumo_agua["año"].unique(), "\n")

columnas_a_limpiar = ['consumo_m3', 'precipitacion_mm', 'ingreso_promedio']

#2. Datos de entrenamiento [2020-2023]
train = df_consumo_agua[(df_consumo_agua["año"] < 2024)]


#3. Datos de prueba [2024]
test = df_consumo_agua[df_consumo_agua["año"] == 2024]


#4. Variables de entrenamiento [2022-2023]
X_train = train[["poblacion", "ingreso_promedio", "temperatura_promedio", "precipitacion_mm"]]
y_train = train["consumo_m3"]

#5. Variables de prueba [2024]
X_test = test[["poblacion", "ingreso_promedio", "temperatura_promedio", "precipitacion_mm"]]
y_test = test["consumo_m3"]

#6. Creacion y entrenamiento del modelo
modelo = LinearRegression()
modelo.fit(X_train, y_train)

#7. Predecir sobre datos de prueba (año 2024)
y_pred = modelo.predict(X_test)

#8. Indicadores de acertividad del modelo
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("  --- Resultados del acertividad del modelo(Predicción 2024) --- ")
print(f"Error Medio Absoluto (MAE): {mae:.2f}")
print(f"Coeficiente de Determinación (R²): {r2:.3f}\n")

#9. Comparar valores reales vs predichos (muestra)
comparacion = pd.DataFrame({
    "Comuna": test["comuna"].values,
    "Año": test["año"].values,
    "Mes": test["mes"].values,
    "Real": y_test.values,
    "Predicho": y_pred.round(2)
})

print("  --- Tabla de comparacion de datos predictivos v/s datos reales (año 2024) ---\n")
print(comparacion.head(20))



#9 Cofecientes del modelo
coeficientes = pd.DataFrame({
    "Variable": X_train.columns,
    "Coeficiente": modelo.coef_.round(4)
})
print("\nCoeficientes del modelo:\n", coeficientes)
