"""
Interfaz de comandos para usar los modelos de aprendizaje NO supervisado.
Sistema Metro Medellin - Actividad de Machine Learning No Supervisado
"""

import sys
from data.dataset_generator_unsupervised import GeneradorDatasetNoSupervisado
from models.clustering_model import ModeloClustering
from models.deteccion_anomalias import ModeloDeteccionAnomalias
from models.reduccion_dimensional import ModeloReduccionDimensional

def comando_entrenar_todos():
    """Entrena todos los modelos no supervisados"""
    print("\n" + "="*60)
    print("ENTRENANDO TODOS LOS MODELOS NO SUPERVISADOS")
    print("="*60)
    
    generador = GeneradorDatasetNoSupervisado()
    df_estaciones = generador.generar_dataset_estaciones()
    df_viajes = generador.generar_dataset_viajes_sin_etiquetas()
    df_perfiles = generador.generar_dataset_perfiles_usuario()
    
    print("\n--- CLUSTERING ---")
    modelo1 = ModeloClustering()
    modelo1.entrenar(df_estaciones, tipo="estaciones")
    
    print("\n--- DETECCION ANOMALIAS ---")
    modelo2 = ModeloDeteccionAnomalias()
    modelo2.entrenar(df_estaciones, tipo="estaciones")
    
    print("\n--- REDUCCION DIMENSIONAL ---")
    modelo3 = ModeloReduccionDimensional()
    modelo3.analisis_completo(df_estaciones, tipo="estaciones")
    
    print("\n¡Modelos no supervisados entrenados!")

def comando_clustering():
    """Aplica clustering a datos"""
    print("\n--- CLUSTERING ---")
    tipo = input("Tipo (estaciones/viajes/perfiles): ").lower()
    
    generador = GeneradorDatasetNoSupervisado()
    
    if tipo == "estaciones":
        df = generador.generar_dataset_estaciones()
    elif tipo == "viajes":
        df = generador.generar_dataset_viajes_sin_etiquetas()
    else:
        df = generador.generar_dataset_perfiles_usuario()
    
    modelo = ModeloClustering()
    modelo.entrenar(df, tipo=tipo)
    modelo.guardar_modelo(f"models/clustering_{tipo}.pkl")

def comando_anomalias():
    """Detecta anomalias en datos"""
    print("\n--- DETECCION DE ANOMALIAS ---")
    tipo = input("Tipo (estaciones/viajes): ").lower()
    
    generador = GeneradorDatasetNoSupervisado()
    
    if tipo == "estaciones":
        df = generador.generar_dataset_estaciones()
    else:
        df = generador.generar_dataset_viajes_sin_etiquetas()
    
    modelo = ModeloDeteccionAnomalias()
    modelo.entrenar(df, tipo=tipo)
    modelo.guardar_modelo(f"models/deteccion_anomalias_{tipo}.pkl")

def comando_reduccion():
    """Aplica reduccion dimensional"""
    print("\n--- REDUCCION DIMENSIONAL ---")
    tipo = input("Tipo (estaciones/viajes): ").lower()
    
    generador = GeneradorDatasetNoSupervisado()
    
    if tipo == "estaciones":
        df = generador.generar_dataset_estaciones()
    else:
        df = generador.generar_dataset_viajes_sin_etiquetas()
    
    modelo = ModeloReduccionDimensional()
    modelo.analisis_completo(df, tipo=tipo)
    modelo.guardar_modelo(f"models/reduccion_dimensional_{tipo}.pkl")

def comando_generar_datasets():
    """Genera datasets no supervisados"""
    print("\n--- GENERANDO DATASETS ---")
    generador = GeneradorDatasetNoSupervisado()
    generador.guardar_datasets()
    print("Datasets guardados en /data")

def mostrar_ayuda():
    print("""
COMANDOS - APRENDIZAJE NO SUPERVISADO
======================================

python comando_unsupervised.py entrenar    - Entrenar todos los modelos
python comando_unsupervised.py clustering  - Clustering (agrupamiento)
python comando_unsupervised.py anomalias   - Deteccion de anomalias
python comando_unsupervised.py reduccion  - Reduccion dimensional
python comando_unsupervised.py datasets   - Generar datasets
python comando_unsupervised.py ayuda      - Mostrar esta ayuda

python main_unsupervised.py             - Ejecutar todo automaticamente
""")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        mostrar_ayuda()
    else:
        comando = sys.argv[1].lower()
        
        if comando == "entrenar":
            comando_entrenar_todos()
        elif comando == "clustering":
            comando_clustering()
        elif comando == "anomalias":
            comando_anomalias()
        elif comando == "reduccion":
            comando_reduccion()
        elif comando == "datasets":
            comando_generar_datasets()
        elif comando == "ayuda":
            mostrar_ayuda()
        else:
            print(f"Comando desconocido: {comando}")
            mostrar_ayuda()
