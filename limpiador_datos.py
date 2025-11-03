import pandas as pd
from pathlib import Path


# 1. Primera lectura del csv
def leer_csv(ruta,sep=","):
    ruta = Path(ruta)
    
    if not ruta.exists():
        return False, None, f"El archivo no existe. {ruta}"
    if ruta.stat().st_size == 0:
        return False, None, "El archivo esta vacio."
    
    for dec in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            df = pd.read_csv(ruta, encoding = dec, sep=sep, low_memory=False)
            return True, df, f"archivo leido correctamente"
        except UnicodeDecodeError:
            continue
        except Exception as e:
            return False, None, f"El archivo no pudo ser leido {e}"
    
    return False, None, f"El archivo no pudo ser procesado"



#2. Estandarizar columnas y eliminar filas duplicadas
def limpiar_filas(df):
    df_temporal = df.copy()
    
    # Normalizacion de nombres de columnas (minusculas y sin espacios)
    df_temporal.columns = df_temporal.columns.str.lower().str.strip()
    
    # Eliminar los duplicados
    antes = len(df_temporal)
    df_temporal = df_temporal.drop_duplicates()
    despues = len(df_temporal)
    
    print(f"\n2.LIMPIEZA: Limpieza de filas hecha. Filas duplicadas eliminadas: {antes - despues}")
    
    return df_temporal

#3. Conversion de tipos de datos por columna
def convertir_tipos_datos(df):
    df_temporal = df.copy()
    
    
    # Columnas de entero: Los no validos se convierten a NaN
    df_temporal['año'] = pd.to_numeric(df_temporal['año'], errors='coerce').astype('Int64')
    df_temporal['poblacion'] = pd.to_numeric(df_temporal['poblacion'], errors='coerce').astype('Int64')
    df_temporal['mes'] = pd.to_numeric(df_temporal['mes'], errors='coerce').astype('Int64')
    
    # Columnas de floats: Los no validos se convierten a NaN
    col_float = ['ingreso_promedio', 'temperatura_promedio', 'precipitacion_mm', 'consumo_m3']
    for col in col_float:
        df_temporal[col] = pd.to_numeric(df_temporal[col], errors='coerce').astype('Float64')
        
    # Columna de texto: Los no validos se convierten a cadena vacia
    col_texto = ['comuna', 'region']
    for col in col_texto:
        df_temporal[col] = df_temporal[col].astype(str).str.upper().str.strip()
        df_temporal[col] = df_temporal[col].replace('NAN','')
        
    print("\n3.CONVERSION DATOS: Columnas con sus tipos normalizadas.")
    
    return df_temporal

# 4. Tratar valores inconsistentes
def valores_inconsistentes(df):
    df_temporal = df.copy()
    
    # Eliminar filas con poblacion o Consumo_m3 nulos
    fecha_antes = len(df_temporal)
    df_temporal = df_temporal.dropna(subset=['poblacion', 'consumo_m3'])
    fecha_despues = len(df_temporal)
    
    # Filtra y elimina filas donde mes está fuera del rango 1 y 12
    mes_antes = len(df_temporal)
    df_temporal = df_temporal.loc[df_temporal['mes'].between(1, 12, inclusive='both')].copy()
    mes_despues = len(df_temporal)
    
    print("\n4.VALORES INCONSISTENTES: Tratamiento de valores inconsistentes realizado con exito.")
    print(f"Se eliminaron {fecha_antes - fecha_despues} filas con Población/Consumo Nulo.")
    print(f"Se eliminaron {mes_antes - mes_despues} filas con mes fuera de [1, 12].\n")
    
    return df_temporal

#5. Encapsulamiento de funciones
def limpiador_csv(ruta, sep=','):
    
    print(f"       ||| INICIO DE PROCESO DE LIMPIEZA PARA: {ruta} |||")
    
    #1
    succes,df,message = leer_csv(ruta,sep)
    if not succes:
        return False, None, f"Error en la lectura: {message}"
    
    print(f"\n1.LECTURA: {message}. Filas iniciales: {len(df)}")
    
    # 2.
    df_limpio = limpiar_filas(df)
    
    # 3. 
    df_limpio = convertir_tipos_datos(df_limpio)
    
    # 4. 
    df_limpio = valores_inconsistentes(df_limpio)
    

    return True, df_limpio, "Proceso de limpieza completado con éxito."
