import os
import pandas as pd
from data.dataset_generator import GeneradorDataset
from models.prediccion_tiempo import ModeloPrediccionTiempo
from models.prediccion_demanda import ModeloPrediccionDemanda
from models.clasificador_transporte import ClasificadorTransporte

def generar_datasets():
    print("\n" + "=" * 70)
    print("FASE 1: GENERACIÓN DE DATASETS")
    print("=" * 70)
    
    generador = GeneradorDataset()
    
    print("\nGenerando datasets históricos...")
    df_viajes = generador.generar_dataset_viajes(n_muestras=5000)
    df_afluencia = generador.generar_dataset_afluencia(n_muestras=3000)
    df_transporte = generador.generar_dataset_transporte(n_muestras=4000)
    
    os.makedirs("data", exist_ok=True)
    
    df_viajes.to_csv("data/dataset_viajes.csv", index=False)
    df_afluencia.to_csv("data/dataset_afluencia.csv", index=False)
    df_transporte.to_csv("data/dataset_transporte.csv", index=False)
    
    print(f"\nDatasets guardados:")
    print(f"  - dataset_viajes.csv: {len(df_viajes)} registros")
    print(f"  - dataset_afluencia.csv: {len(df_afluencia)} registros")
    print(f"  - dataset_transporte.csv: {len(df_transporte)} registros")
    
    return df_viajes, df_afluencia, df_transporte

def entrenar_modelo_tiempo(df_viajes):
    print("\n" + "=" * 70)
    print("FASE 2: MODELO DE PREDICCIÓN DE TIEMPO DE VIAJE")
    print("=" * 70)
    
    modelo = ModeloPrediccionTiempo()
    modelo.entrenar(df_viajes, modelo_tipo="random_forest")
    modelo.guardar_modelo("models/tiempo_viaje_model.pkl")
    
    print("\n" + "-" * 50)
    print("EJEMPLO DE PREDICCIÓN")
    print("-" * 50)
    
    ejemplos = [
        {
            "origen": "San Antonio", "destino": "Poblado",
            "tipo_transporte": "Metro", "distancia_km": 5.0,
            "dia_semana": "Lunes", "hora": 8,
            "es_hora_pico": 1, "es_finde_semana": 0,
            "transferencias": 0, "clima": "Soleado"
        },
        {
            "origen": "Niquia", "destino": "La Estrella",
            "tipo_transporte": "Metro", "distancia_km": 18.0,
            "dia_semana": "Sabado", "hora": 14,
            "es_hora_pico": 0, "es_finde_semana": 1,
            "transferencias": 1, "clima": "Nublado"
        }
    ]
    
    for i, ejemplo in enumerate(ejemplos, 1):
        tiempo = modelo.predecir(ejemplo)
        print(f"\n  Viaje {i}: {ejemplo['origen']} -> {ejemplo['destino']}")
        print(f"    Transporte: {ejemplo['tipo_transporte']}")
        print(f"    Día: {ejemplo['dia_semana']}, Hora: {ejemplo['hora']}:00")
        print(f"    Tiempo predicho: {tiempo:.1f} minutos")
    
    return modelo

def entrenar_modelo_demanda(df_afluencia):
    print("\n" + "=" * 70)
    print("FASE 3: MODELO DE PREDICCIÓN DE DEMANDA")
    print("=" * 70)
    
    modelo = ModeloPrediccionDemanda()
    modelo.entrenar(df_afluencia, modelo_tipo="random_forest")
    modelo.guardar_modelo("models/demanda_model.pkl")
    
    print("\n" + "-" * 50)
    print("EJEMPLO DE PREDICCIÓN")
    print("-" * 50)
    
    ejemplos = [
        {
            "estacion": "San Antonio", "dia_semana": "Martes",
            "hora": 8, "es_hora_pico": 1, "es_finde_semana": 0,
            "clima": "Soleado", "evento_especial": 0
        },
        {
            "estacion": "Poblado", "dia_semana": "Domingo",
            "hora": 12, "es_hora_pico": 0, "es_finde_semana": 1,
            "clima": "Soleado", "evento_especial": 0
        }
    ]
    
    for i, ejemplo in enumerate(ejemplos, 1):
        resultado = modelo.predecir_ocupacion(ejemplo)
        print(f"\n  Estación {i}: {ejemplo['estacion']}")
        print(f"    Día: {ejemplo['dia_semana']}, Hora: {ejemplo['hora']}:00")
        print(f"    Afluencia predicha: {resultado['afluencia_predicha']} pasajeros")
        print(f"    Ocupación: {resultado['porcentaje_ocupacion']}% ({resultado['nivel_congestion']})")
    
    return modelo

def entrenar_clasificador_transporte(df_transporte):
    print("\n" + "=" * 70)
    print("FASE 4: CLASIFICADOR DE TRANSPORTE ÓPTIMO")
    print("=" * 70)
    
    modelo = ClasificadorTransporte()
    modelo.entrenar(df_transporte, modelo_tipo="random_forest")
    modelo.guardar_modelo("models/clasificador_transporte_model.pkl")
    
    print("\n" + "-" * 50)
    print("EJEMPLO DE PREDICCIÓN")
    print("-" * 50)
    
    ejemplos = [
        {
            "origen": "San Antonio", "destino": "Poblado",
            "distancia_km": 5.0, "dia_semana": "Viernes",
            "hora": 18, "es_hora_pico": 1, "es_finde_semana": 0,
            "es_urgente": 0, "presupuesto": "medio"
        },
        {
            "origen": "Bello", "destino": "Envigado",
            "distancia_km": 12.0, "dia_semana": "Domingo",
            "hora": 10, "es_hora_pico": 0, "es_finde_semana": 1,
            "es_urgente": 0, "presupuesto": "bajo"
        }
    ]
    
    for i, ejemplo in enumerate(ejemplos, 1):
        resultado = modelo.predecir(ejemplo)
        print(f"\n  Viaje {i}: {ejemplo['origen']} -> {ejemplo['destino']}")
        print(f"    Distancia: {ejemplo['distancia_km']} km, Presupuesto: {ejemplo['presupuesto']}")
        print(f"    Transporte recomendado: {resultado['transporte_predicho']}")
        print("    Probabilidades:")
        for t, p in sorted(resultado['probabilidades'].items(), key=lambda x: x[1], reverse=True):
            print(f"      {t}: {p*100:.1f}%")
    
    return modelo

def mostrar_resumen():
    print("\n" + "=" * 70)
    print("RESUMEN: MODELOS DE APRENDIZAJE SUPERVISADO IMPLEMENTADOS")
    print("=" * 70)
    
    print("""
    1. PREDICCIÓN DE TIEMPO DE VIAJE (Regresión)
       - Objetivo: Estimar el tiempo de viaje en minutos
       - Algoritmo: Random Forest Regressor
       - Métricas: MAE, RMSE, R²
       - Características: distancia, hora, tipo transporte, clima, día
       
    2. PREDICCIÓN DE DEMANDA (Regresión)
       - Objetivo: Estimar la afluencia de pasajeros
       - Algoritmo: Random Forest Regressor
       - Métricas: MAE, RMSE, R²
       - Características: estación, hora, día, clima, eventos
       
    3. CLASIFICADOR DE TRANSPORTE ÓPTIMO (Clasificación)
       - Objetivo: Recomendar el mejor tipo de transporte
       - Algoritmo: Random Forest Classifier
       - Métricas: Accuracy, Precision, Recall, F1-Score
       - Clases: Metro, Bus, Taxi, Cable, Caminando
       
    ARCHIVOS GENERADOS:
       - data/dataset_viajes.csv
       - data/dataset_afluencia.csv
       - data/dataset_transporte.csv
       - models/tiempo_viaje_model.pkl
       - models/demanda_model.pkl
       - models/clasificador_transporte_model.pkl
    """)

def main():
    print("\n" + "=" * 70)
    print("SISTEMA DE APRENDIZAJE SUPERVISADO - METRO MEDELLÍN")
    print("=" * 70)
    
    df_viajes, df_afluencia, df_transporte = generar_datasets()
    
    modelo_tiempo = entrenar_modelo_tiempo(df_viajes)
    
    modelo_demanda = entrenar_modelo_demanda(df_afluencia)
    
    modelo_transporte = entrenar_clasificador_transporte(df_transporte)
    
    mostrar_resumen()
    
    print("\n" + "=" * 70)
    print("ENTRENAMIENTO COMPLETADO EXITOSAMENTE")
    print("=" * 70 + "\n")

if __name__ == "__main__":
    main()
