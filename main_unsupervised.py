import os
import pandas as pd
from data.dataset_generator_unsupervised import GeneradorDatasetNoSupervisado
from models.clustering_model import ModeloClustering
from models.deteccion_anomalias import ModeloDeteccionAnomalias
from models.reduccion_dimensional import ModeloReduccionDimensional

def generar_datasets():
    print("\n" + "=" * 70)
    print("FASE 1: GENERACION DE DATASETS (APRENDIZAJE NO SUPERVISADO)")
    print("=" * 70)
    
    generador = GeneradorDatasetNoSupervisado()
    
    df_estaciones = generador.generar_dataset_estaciones(n_muestras=1000)
    df_viajes = generador.generar_dataset_viajes_sin_etiquetas(n_muestras=1500)
    df_perfiles = generador.generar_dataset_perfiles_usuario(n_muestras=800)
    
    os.makedirs("data", exist_ok=True)
    
    df_estaciones.to_csv("data/dataset_clustering_estaciones.csv", index=False)
    df_viajes.to_csv("data/dataset_clustering_viajes.csv", index=False)
    df_perfiles.to_csv("data/dataset_perfiles_usuario.csv", index=False)
    
    print(f"\nDatasets guardados:")
    print(f"  - dataset_clustering_estaciones.csv: {len(df_estaciones)} registros")
    print(f"  - dataset_clustering_viajes.csv: {len(df_viajes)} registros")
    print(f"  - dataset_perfiles_usuario.csv: {len(df_perfiles)} registros")
    
    return df_estaciones, df_viajes, df_perfiles

def entrenar_clustering(df_estaciones, df_viajes, df_perfiles):
    print("\n" + "=" * 70)
    print("FASE 2: CLUSTERING (AGRUPAMIENTO)")
    print("=" * 70)
    
    modelo_est = ModeloClustering()
    modelo_est.entrenar(df_estaciones, tipo="estaciones")
    modelo_est.guardar_modelo("models/clustering_estaciones.pkl")
    
    modelo_viajes = ModeloClustering()
    modelo_viajes.entrenar(df_viajes, tipo="viajes")
    modelo_viajes.guardar_modelo("models/clustering_viajes.pkl")
    
    modelo_perfiles = ModeloClustering()
    modelo_perfiles.entrenar(df_perfiles, tipo="perfiles")
    modelo_perfiles.guardar_modelo("models/clustering_perfiles.pkl")
    
    return modelo_est, modelo_viajes, modelo_perfiles

def entrenar_deteccion_anomalias(df_estaciones, df_viajes):
    print("\n" + "=" * 70)
    print("FASE 3: DETECCION DE ANOMALIAS")
    print("=" * 70)
    
    modelo_est = ModeloDeteccionAnomalias()
    modelo_est.entrenar(df_estaciones, tipo="estaciones", algoritmo="isolation_forest")
    modelo_est.guardar_modelo("models/deteccion_anomalias_estaciones.pkl")
    
    modelo_viajes = ModeloDeteccionAnomalias()
    modelo_viajes.entrenar(df_viajes, tipo="viajes", algoritmo="isolation_forest")
    modelo_viajes.guardar_modelo("models/deteccion_anomalias_viajes.pkl")
    
    return modelo_est, modelo_viajes

def entrenar_reduccion_dimensional(df_estaciones, df_viajes):
    print("\n" + "=" * 70)
    print("FASE 4: REDUCCION DIMENSIONAL")
    print("=" * 70)
    
    modelo_est = ModeloReduccionDimensional()
    modelo_est.analisis_completo(df_estaciones, tipo="estaciones")
    modelo_est.guardar_modelo("models/reduccion_dimensional_estaciones.pkl")
    
    modelo_viajes = ModeloReduccionDimensional()
    modelo_viajes.analisis_completo(df_viajes, tipo="viajes")
    modelo_viajes.guardar_modelo("models/reduccion_dimensional_viajes.pkl")
    
    return modelo_est, modelo_viajes

def mostrar_resumen():
    print("\n" + "=" * 70)
    print("RESUMEN: MODELOS DE APRENDIZAJE NO SUPERVISADO")
    print("=" * 70)
    
    print("""
    1. CLUSTERING (Agrupamiento - K-Means)
       - Objetivo: Agrupar datos similares sin etiquetas
       - Algoritmo: K-Means
       - Datasets: Estaciones, Viajes, Perfiles de usuario
       - Metricas: Silhouette, Calinski-Harabasz, Davies-Bouldin
       
    2. DETECCION DE ANOMALIAS
       - Objetivo: Identificar patrones anormales o inusuales
       - Algoritmos: Isolation Forest, LOF, One-Class SVM
       - Datasets: Estaciones, Viajes
       - Uso: Deteccion de horarios problematicos, fallas del sistema
       
    3. REDUCCION DIMENSIONAL
       - Objetivo: Reducir caracteristicas manteniendo informacion
       - Metodos: PCA, t-SNE
       - Uso: Visualizacion y descubrimiento de patrones ocultos
       
    ARCHIVOS GENERADOS:
       - data/dataset_clustering_*.csv
       - models/clustering_*.pkl
       - models/deteccion_anomalias_*.pkl
       - models/reduccion_dimensional_*.pkl
    """)

def main():
    print("\n" + "=" * 70)
    print("APRENDIZAJE NO SUPERVISADO - METRO MEDELLIN")
    print("=" * 70)
    
    df_estaciones, df_viajes, df_perfiles = generar_datasets()
    
    modelos_clustering = entrenar_clustering(df_estaciones, df_viajes, df_perfiles)
    
    modelos_anomalias = entrenar_deteccion_anomalias(df_estaciones, df_viajes)
    
    modelos_reduccion = entrenar_reduccion_dimensional(df_estaciones, df_viajes)
    
    mostrar_resumen()
    
    print("\n" + "=" * 70)
    print("ENTRENAMIENTO COMPLETADO EXITOSAMENTE")
    print("=" * 70 + "\n")

if __name__ == "__main__":
    main()
