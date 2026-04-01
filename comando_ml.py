"""
Interfaz de comandos para usar los modelos de aprendizaje automático.
Sistema Metro Medellín - Actividad de Machine Learning
"""

import sys
from data.dataset_generator import GeneradorDataset
from models.prediccion_tiempo import ModeloPrediccionTiempo
from models.prediccion_demanda import ModeloPrediccionDemanda
from models.clasificador_transporte import ClasificadorTransporte

def comando_entrenar_todos():
    """Entrena todos los modelos de ML"""
    print("\n" + "="*60)
    print("ENTRENANDO TODOS LOS MODELOS")
    print("="*60)
    
    generador = GeneradorDataset()
    df_viajes = generador.generar_dataset_viajes()
    df_afluencia = generador.generar_dataset_afluencia()
    df_transporte = generador.generar_dataset_transporte()
    
    modelo_tiempo = ModeloPrediccionTiempo()
    modelo_tiempo.entrenar(df_viajes)
    modelo_tiempo.guardar_modelo()
    
    modelo_demanda = ModeloPrediccionDemanda()
    modelo_demanda.entrenar(df_afluencia)
    modelo_demanda.guardar_modelo()
    
    modelo_transporte = ClasificadorTransporte()
    modelo_transporte.entrenar(df_transporte)
    modelo_transporte.guardar_modelo()
    
    print("\n¡Todos los modelos entrenados exitosamente!")

def comando_predecir_tiempo():
    """Predice tiempo de viaje"""
    print("\n--- PREDICCIÓN DE TIEMPO DE VIAJE ---")
    
    origen = input("Origen: ")
    destino = input("Destino: ")
    tipo = input("Tipo transporte (Metro/Bus/Taxi): ")
    distancia = float(input("Distancia (km): "))
    dia = input("Día de semana: ")
    hora = int(input("Hora (0-23): "))
    clima = input("Clima (Soleado/Nublado/Lluvia/Tormenta): ")
    
    modelo = ModeloPrediccionTiempo()
    try:
        modelo.cargar_modelo("models/tiempo_viaje_model.pkl")
    except:
        print("Modelo no encontrado. Entrenando...")
        generador = GeneradorDataset()
        modelo.entrenar(generador.generar_dataset_viajes())
    
    datos = {
        "origen": origen, "destino": destino, "tipo_transporte": tipo,
        "distancia_km": distancia, "dia_semana": dia, "hora": hora,
        "es_hora_pico": 1 if hora in [6,7,8,11,12,13,17,18,19] else 0,
        "es_finde_semana": 1 if dia in ["Sabado","Domingo"] else 0,
        "transferencias": 0, "clima": clima
    }
    
    tiempo = modelo.predecir(datos)
    print(f"\nTiempo estimado: {tiempo:.1f} minutos")

def comando_predecir_demanda():
    """Predice demanda de pasajeros"""
    print("\n--- PREDICCIÓN DE DEMANDA ---")
    
    estacion = input("Estación: ")
    dia = input("Día de semana: ")
    hora = int(input("Hora (0-23): "))
    clima = input("Clima (Soleado/Nublado/Lluvia/Tormenta): ")
    evento = input("¿Evento especial? (s/n): ")
    
    modelo = ModeloPrediccionDemanda()
    try:
        modelo.cargar_modelo("models/demanda_model.pkl")
    except:
        print("Modelo no encontrado. Entrenando...")
        generador = GeneradorDataset()
        modelo.entrenar(generador.generar_dataset_afluencia())
    
    datos = {
        "estacion": estacion, "dia_semana": dia, "hora": hora,
        "es_hora_pico": 1 if hora in [6,7,8,11,12,13,17,18,19] else 0,
        "es_finde_semana": 1 if dia in ["Sabado","Domingo"] else 0,
        "clima": clima, "evento_especial": 1 if evento.lower() == "s" else 0
    }
    
    resultado = modelo.predecir_ocupacion(datos)
    print(f"\nAfluencia predicha: {resultado['afluencia_predicha']} pasajeros")
    print(f"Ocupación: {resultado['porcentaje_ocupacion']}%")
    print(f"Nivel congestión: {resultado['nivel_congestion']}")

def comando_recomendar_transporte():
    """Recomienda tipo de transporte óptimo"""
    print("\n--- RECOMENDACIÓN DE TRANSPORTE ---")
    
    origen = input("Origen: ")
    destino = input("Destino: ")
    distancia = float(input("Distancia (km): "))
    dia = input("Día de semana: ")
    hora = int(input("Hora (0-23): "))
    presupuesto = input("Presupuesto (bajo/medio/alto): ")
    urgente = input("¿Es urgente? (s/n): ")
    
    modelo = ClasificadorTransporte()
    try:
        modelo.cargar_modelo("models/clasificador_transporte_model.pkl")
    except:
        print("Modelo no encontrado. Entrenando...")
        generador = GeneradorDataset()
        modelo.entrenar(generador.generar_dataset_transporte())
    
    datos = {
        "origen": origen, "destino": destino, "distancia_km": distancia,
        "dia_semana": dia, "hora": hora,
        "es_hora_pico": 1 if hora in [6,7,8,11,12,13,17,18,19] else 0,
        "es_finde_semana": 1 if dia in ["Sabado","Domingo"] else 0,
        "es_urgente": 1 if urgente.lower() == "s" else 0,
        "presupuesto": presupuesto
    }
    
    resultado = modelo.predecir(datos)
    print(f"\nTransporte recomendado: {resultado['transporte_predicho']}")
    print("\nProbabilidades:")
    for t, p in sorted(resultado['probabilidades'].items(), key=lambda x: x[1], reverse=True):
        print(f"  {t}: {p*100:.1f}%")

def comando_generar_datasets():
    """Genera datasets de ejemplo"""
    print("\n--- GENERANDO DATASETS ---")
    generador = GeneradorDataset()
    generador.guardar_datasets()
    print("Datasets guardados en /data")

def mostrar_ayuda():
    print("""
COMANDOS - SISTEMA ML METRO MEDELLIN
====================================

python comando_ml.py entrenar    - Entrena todos los modelos ML
python comando_ml.py predecir    - Predecir tiempo de viaje
python comando_ml.py demanda     - Predecir demanda de pasajeros
python comando_ml.py recomendar  - Recomendar transporte optimo
python comando_ml.py datasets    - Generar datasets
python comando_ml.py ayuda       - Mostrar esta ayuda

python main_ml.py               - Ejecutar todo automaticamente
""")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        mostrar_ayuda()
    else:
        comando = sys.argv[1].lower()
        
        if comando == "entrenar":
            comando_entrenar_todos()
        elif comando == "predecir":
            comando_predecir_tiempo()
        elif comando == "demanda":
            comando_predecir_demanda()
        elif comando == "recomendar":
            comando_recomendar_transporte()
        elif comando == "datasets":
            comando_generar_datasets()
        elif comando == "ayuda":
            mostrar_ayuda()
        else:
            print(f"Comando desconocido: {comando}")
            mostrar_ayuda()
