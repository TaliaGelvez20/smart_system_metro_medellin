"""
AYUDA GENERAL - Sistema Metro Medellin
Muestra todos los comandos disponibles del proyecto
"""

def mostrar_ayuda_general():
    print("""
================================================================================
           SISTEMA METRO MEDELLIN - COMANDOS DISPONIBLES           
================================================================================

SISTEMA DE BUSQUEDA (Actividad 1)
----------------------------------
  python main.py              Ejecutar busqueda Dijkstra (interactivo)

APRENDIZAJE SUPERVISADO (Actividad 2)
---------------------------------------
  python main_ml.py           Ejecutar todos los modelos supervisados
  
  python comando_ml.py datasets     Generar datasets
  python comando_ml.py entrenar     Entrenar modelos
  python comando_ml.py predecir     Predecir tiempo de viaje
  python comando_ml.py demanda      Predecir demanda pasajeros
  python comando_ml.py recomendar   Recomendar transporte optimo
  python comando_ml.py ayuda        Mostrar ayuda ML supervisado

APRENDIZAJE NO SUPERVISADO (Actividad 3)
-----------------------------------------
  python main_unsupervised.py        Ejecutar todos los modelos no supervisados
  
  python comando_unsupervised.py datasets   Generar datasets
  python comando_unsupervised.py entrenar  Entrenar modelos
  python comando_unsupervised.py clustering Clustering (K-Means)
  python comando_unsupervised.py anomalias  Deteccion anomalias
  python comando_unsupervised.py reduccion Reduccion dimensional
  python comando_unsupervised.py ayuda     Mostrar ayuda ML no supervisado

AYUDA GENERAL
-------------
  python help.py              Mostrar esta ayuda general

================================================================================
                      RESUMEN RAPIDO                               
================================================================================
  main.py                -> Busqueda Dijkstra                      
  main_ml.py             -> ML Supervisado completo               
  main_unsupervised.py   -> ML No Supervisado completo             
  help.py                -> Esta ayuda                            
================================================================================
""")

if __name__ == "__main__":
    mostrar_ayuda_general()
