# Sistema Inteligente de Transporte Valle de Aburrá

Este proyecto implementa un **sistema inteligente de transporte** para el Valle de Aburrá (Medellín, Colombia), diseñado como una actividad académica en inteligencia artificial. Utiliza un enfoque basado en **búsqueda y sistemas de reglas lógicas** para encontrar la mejor ruta entre dos puntos en el sistema de transporte masivo, considerando tiempo, costo económico y eficiencia. Además, incluye **modelos de aprendizaje automático supervisado** para predicción de tiempos, demanda y clasificación de transporte óptimo.

## Funcionalidad

El sistema modela la red de transporte como un **grafo multimodal**, donde:
- **Nodos**: Estaciones y puntos de interés (ej. estaciones de Metro, terminales de buses, centros comerciales).
- **Aristas**: Conexiones entre nodos con atributos como tipo de transporte (Metro, Bus, Taxi, Cable, Caminando), tiempo de viaje (minutos) y costo monetario (pesos colombianos).

### Componentes Principales - Búsqueda y Reglas
- **SistemaTransporte** (`core/sistema_transporte.py`): Gestiona el grafo, agregando estaciones y conexiones.
- **MotorBusqueda** (`core/motor_busqueda.py`): Implementa el algoritmo de Dijkstra para encontrar la ruta más corta en términos de tiempo.
- **SistemaInteligente** (`ai/sistema_inteligente.py`): Aplica reglas lógicas inteligentes, como:
  - Penalización por transbordos (cambios de tipo de transporte).
  - Preferencia por Metro (más eficiente).
  - Desincentivo al uso de taxis (costos altos).
  - Cálculo de costo real basado en tarifas por tipo de transporte usado.
- **Setup** (`data/setup.py`): Configura el grafo con datos reales del Metro de Medellín (Líneas A y B), buses integrados y conexiones adicionales.

### Componentes Principales - Aprendizaje Automático
- **GeneradorDataset** (`data/dataset_generator.py`): Genera datasets históricos simulados de viajes, demanda y transporte.
- **ModeloPrediccionTiempo** (`models/prediccion_tiempo.py`): Modelo de regresión para predecir tiempo de viaje.
- **ModeloPrediccionDemanda** (`models/prediccion_demanda.py`): Modelo de regresión para predecir afluencia de pasajeros.
- **ClasificadorTransporte** (`models/clasificador_transporte.py`): Modelo de clasificación para recomendar tipo de transporte óptimo.

### Algoritmo
- **Base**: Dijkstra para caminos más cortos (optimizado por tiempo).
- **Machine Learning**: Random Forest para regresión y clasificación.

## Requisitos
- Python 3.7 o superior.
- Bibliotecas para ML: `scikit-learn`, `pandas`, `numpy`, `joblib`

```bash
pip install scikit-learn pandas numpy joblib
```

## Cómo Ejecutar el Proyecto

### Sistema de Búsqueda (Actividad Anterior)
```bash
python main.py
```

### Sistema de Aprendizaje Automático (Actividad Actual)

```bash
# Ejecución completa de todos los modelos ML
python main_ml.py

# Generación de datasets
python comando_ml.py datasets

# Entrenar todos los modelos
python comando_ml.py entrenar

# Predicción de tiempo de viaje (interactivo)
python comando_ml.py predecir

# Predicción de demanda de pasajeros (interactivo)
python comando_ml.py demanda

# Recomendación de transporte óptimo (interactivo)
python comando_ml.py recomendar

# Mostrar ayuda
python comando_ml.py ayuda
```

## Comandos Disponibles

| Comando | Descripción |
|---------|-------------|
| `python main.py` | Sistema de búsqueda de rutas (Dijkstra) |
| `python main_ml.py` | Entrenamiento completo de todos los modelos ML |
| `python comando_ml.py datasets` | Genera datasets históricos |
| `python comando_ml.py entrenar` | Entrena todos los modelos ML |
| `python comando_ml.py predecir` | Predice tiempo de viaje (modo interactivo) |
| `python comando_ml.py demanda` | Predice demanda de pasajeros (modo interactivo) |
| `python comando_ml.py recomendar` | Recomienda transporte óptimo (modo interactivo) |
| `python comando_ml.py ayuda` | Muestra todos los comandos disponibles |

## Modelos de Aprendizaje Supervisado

### 1. Predicción de Tiempo de Viaje (Regresión)
- **Algoritmo**: Random Forest Regressor
- **Objetivo**: Estimar el tiempo de viaje en minutos
- **Características**: distancia, hora, tipo transporte, clima, día, hora pico
- **Métricas**: MAE, RMSE, R²

### 2. Predicción de Demanda (Regresión)
- **Algoritmo**: Random Forest Regressor
- **Objetivo**: Estimar la afluencia de pasajeros por estación
- **Características**: estación, hora, día, clima, eventos especiales
- **Métricas**: MAE, RMSE, R²

### 3. Clasificador de Transporte Óptimo (Clasificación)
- **Algoritmo**: Random Forest Classifier
- **Objetivo**: Recomendar el mejor tipo de transporte
- **Clases**: Metro, Bus, Taxi, Cable, Caminando
- **Métricas**: Accuracy, Precision, Recall, F1-Score

## Datasets Generados

| Dataset | Registros | Descripción |
|---------|-----------|-------------|
| `dataset_viajes.csv` | 5000 | Historial de viajes con tiempo, costo y tipo |
| `dataset_afluencia.csv` | 3000 | Datos de demanda por estación y hora |
| `dataset_transporte.csv` | 4000 | Recomendaciones de transporte óptimo |

## Archivos de Modelos Entrenados

| Modelo | Archivo |
|--------|---------|
| Predicción Tiempo | `models/tiempo_viaje_model.pkl` |
| Predicción Demanda | `models/demanda_model.pkl` |
| Clasificador Transporte | `models/clasificador_transporte_model.pkl` |

## Ejemplo de Salida - Sistema de Búsqueda
```
=== Sistema Inteligente Transporte Valle de Aburrá ===

Origen: Niquia
Destino: Poblado

Ruta encontrada:
Niquia -> Bello -> Madera -> Acevedo -> ... -> Poblado

Detalle de la ruta:
Niquia -> Bello [Metro]
Bello -> Madera [Metro]
...
Poblado [Metro]

Tiempo total: 45 min
Transbordos: 0
Costo real: 3200
```

## Cómo Extender el Proyecto

### 1. **Agregar Más Estaciones y Conexiones**
   - Edita `data/setup.py`:
     - Agrega nuevas estaciones llamando a `sistema.agregar_estacion("NuevaEstacion")`.
     - Conecta estaciones con `sistema.conectar(origen, destino, tipo, tiempo, costo)`.
     - Ejemplo: Agregar una nueva línea de Metro o buses a municipios adicionales.
   - Actualiza tiempos y costos con datos reales para mayor precisión.

### 2. **Agregar Nuevos Tipos de Transporte**
   - En `data/setup.py`, usa tipos como "Bicicleta", "Cable" o "Tranvía".
   - Actualiza las reglas en `ai/sistema_inteligente.py` para incluir costos o penalizaciones.

### 3. **Mejorar las Reglas Inteligentes**
   - Edita `ai/sistema_inteligente.py`:
     - Modifica `analizar_ruta` para contar transbordos de manera más sofisticada.
     - Ajusta `calcular_costo_real` para incluir descuentos o factores ambientales.

### 4. **Integrar Algoritmos Avanzados**
   - Reemplaza Dijkstra en `core/motor_busqueda.py` con A* (agregando heurísticas).
   - Mejora los modelos de ML con más datos o algoritmos diferentes.

### 5. **Interfaz Gráfica o Web**
   - Crea una interfaz con Tkinter, Flask o Streamlit.
   - Agrega mapas interactivos usando Folium.

### 6. **Validación y Pruebas**
   - Agrega pruebas unitarias con `unittest`.
   - Valida con datos reales del Metro de Medellín.

## Estructura del Proyecto

```
smart_system_metro_medellin/
├── main.py                      # Punto de entrada (búsqueda Dijkstra)
├── main_ml.py                   # Punto de entrada (ML - entrenamiento completo)
├── comando_ml.py                # Interfaz de comandos CLI
├── README.md                    # Este archivo
├── ai/
│   └── sistema_inteligente.py   # Lógica inteligente y reglas
├── core/
│   ├── motor_busqueda.py        # Algoritmo de búsqueda (Dijkstra)
│   └── sistema_transporte.py    # Gestión del grafo
├── data/
│   ├── setup.py                 # Configuración del grafo
│   ├── dataset_generator.py     # Generador de datasets ML
│   ├── dataset_viajes.csv       # Dataset de viajes
│   ├── dataset_afluencia.csv    # Dataset de demanda
│   └── dataset_transporte.csv   # Dataset de transporte
└── models/
    ├── conexion.py              # Modelo de conexiones
    ├── prediccion_tiempo.py     # Modelo regresión tiempo
    ├── prediccion_demanda.py    # Modelo regresión demanda
    ├── clasificador_transporte.py # Clasificador transporte
    ├── tiempo_viaje_model.pkl   # Modelo entrenado
    ├── demanda_model.pkl        # Modelo entrenado
    └── clasificador_transporte_model.pkl # Modelo entrenado
```

## Contribución
Si deseas contribuir:
1. Haz un fork del repositorio.
2. Crea una rama para tu feature.
3. Envía un pull request con cambios bien documentados.

Este proyecto es educativo y puede expandirse para aplicaciones reales en planificación urbana o apps de transporte.

¡Disfruta explorando el transporte de Medellín de manera inteligente!
