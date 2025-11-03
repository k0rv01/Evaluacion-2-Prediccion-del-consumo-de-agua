import pandas as pd
from limpiador_datos import limpiador_csv
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import numpy as np


# Ruta del archivo
ruta = 'consumo_agua_comunas.csv'
separador = ','

#Convertir variables categóricas (Comuna, Region) en numéricas
df_procesado = pd.get_dummies(df_consumo_agua, columns=['comuna', 'region'], drop_first=True)

#Definir características y objetivo 
X = df_procesado.drop('consumo_m3', axis=1)
y = df_procesado['consumo_m3']
#Guardamos los nombres de las características para despues poder usarlo
feature_names = X.columns

#Dividir los Datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Random Forest y su entrenamiento
rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)

#Obtener las importancias de las características del modelo entrenado
importances = rf_model.feature_importances_
#Crear una serie de pandas para ver los nombres junto a su importancia
importance_series = pd.Series(importances, index=feature_names)
#Ordenar las importancias de mayor a menor e tomar solo las 5 primeras
sorted_importances = importance_series.sort_values(ascending=False)
top_5_importances = sorted_importances.head(5)
# Convertir a porcentaje y formatear con 2 decimales y el símbolo '%'
top_5_percent = (top_5_importances * 100).map('{:.2f}%'.format)

print("Las 5 variables que mas influyen en el consumo por metro cubico:")
print(top_5_percent.to_string())
