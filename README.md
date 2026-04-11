# Sistema Inteligente de Transporte Valle de Aburrá

Este proyecto implementa un **sistema inteligente de transporte** para el Valle de Aburrá (Medellín, Colombia), diseñado como actividad académica en inteligencia artificial. Utiliza:

- **Búsqueda y sistemas de reglas lógicas** (Algoritmo Dijkstra)
- **Aprendizaje Supervisado** (Predicción y Clasificación)
- **Aprendizaje No Supervisado** (Clustering y Detección de Anomalías)

## Funcionalidad

El sistema modela la red de transporte como un **grafo multimodal**, donde:
- **Nodos**: Estaciones y puntos de interés
- **Aristas**: Conexiones con tipo de transporte, tiempo y costo

## Requisitos
- Python 3.7+
- Bibliotecas: `scikit-learn`, `pandas`, `numpy`, `joblib`

```bash
pip install scikit-learn pandas numpy joblib
```

---

## COMANDOS DISPONIBLES

### Ayuda General
```bash
python help.py                # Mostrar todos los comandos disponibles
```

### Sistema de Búsqueda (Actividad 1)
```bash
python main.py               # Ejecutar busqueda Dijkstra (interactivo)
```

### Actividad 3 - Métodos de aprendizaje supervisado
```bash
python main_ml.py                  # Ejecutar todos los modelos
python comando_ml.py datasets      # Generar datasets
python comando_ml.py entrenar      # Entrenar modelos
python comando_ml.py predecir      # Predecir tiempo de viaje
python comando_ml.py demanda       # Predecir demanda
python comando_ml.py recomendar    # Recomendar transporte
python comando_ml.py ayuda         # Mostrar ayuda ML supervisado
```

### Actividad 4 - Métodos de aprendizaje no supervisado
```bash
python main_unsupervised.py              # Ejecutar todos los modelos
python comando_unsupervised.py datasets   # Generar datasets
python comando_unsupervised.py entrenar   # Entrenar modelos
python comando_unsupervised.py clustering # Clustering (agrupamiento)
python comando_unsupervised.py anomalias  # Deteccion de anomalias
python comando_unsupervised.py reduccion  # Reduccion dimensional
python comando_unsupervised.py ayuda     # Mostrar ayuda ML no supervisado
```

---

## RESUMEN DE COMANDOS

| Comando | Descripción |
|---------|-------------|
| `python help.py` | Mostrar ayuda general (todos los comandos) |
| `python main.py` | Sistema de búsqueda de rutas (Dijkstra) |
| `python main_ml.py` | Aprendizaje supervisado completo |
| `python main_unsupervised.py` | Aprendizaje no supervisado completo |
| `python comando_ml.py predecir` | Predecir tiempo de viaje |
| `python comando_ml.py demanda` | Predecir demanda de pasajeros |
| `python comando_ml.py recomendar` | Recomendar transporte óptimo |
| `python comando_unsupervised.py clustering` | Clustering de datos |
| `python comando_unsupervised.py anomalias` | Detectar anomalías |
| `python comando_unsupervised.py reduccion` | Reducción dimensional |

---

## MODELOS IMPLEMENTADOS

### Aprendizaje Supervisado

| Modelo | Tipo | Algoritmo | Objetivo |
|--------|------|-----------|----------|
| Predicción Tiempo | Regresión | Random Forest | Estimar tiempo de viaje |
| Predicción Demanda | Regresión | Random Forest | Estimar afluencia pasajeros |
| Clasificador Transporte | Clasificación | Random Forest | Recomendar tipo transporte |

### Aprendizaje No Supervisado

| Modelo | Tipo | Algoritmo | Objetivo |
|--------|------|-----------|----------|
| Clustering Estaciones | Agrupamiento | K-Means | Segmentar estaciones por demanda |
| Clustering Viajes | Agrupamiento | K-Means | Identificar patrones de viaje |
| Clustering Perfiles | Agrupamiento | K-Means | Segmentar usuarios |
| Detección Anomalías | Anomalías | Isolation Forest | Identificar casos inusuales |
| Reducción Dimensional | Reducción | PCA, t-SNE | Visualizar y descubrir patrones |

---

## DATSETS GENERADOS

### Aprendizaje Supervisado
| Dataset | Registros |
|---------|-----------|
| `dataset_viajes.csv` | 5000 |
| `dataset_afluencia.csv` | 3000 |
| `dataset_transporte.csv` | 4000 |

### Aprendizaje No Supervisado
| Dataset | Registros |
|---------|-----------|
| `dataset_clustering_estaciones.csv` | 1000 |
| `dataset_clustering_viajes.csv` | 1500 |
| `dataset_perfiles_usuario.csv` | 800 |

---

## RESULTADOS DE ENTRENAMIENTO

### Aprendizaje Supervisado
- **Predicción Tiempo**: R² = 0.833, MAE = 4.94 min
- **Predicción Demanda**: R² = 0.879, MAE = 273.75 pasajeros
- **Clasificador Transporte**: Accuracy = 79.5%

### Aprendizaje No Supervisado
- **Clustering Estaciones**: K=3 clusters, Silhouette = 0.1497
- **Clustering Viajes**: K=9 clusters, Silhouette = 0.2162
- **Clustering Perfiles**: K=6 clusters, Silhouette = 0.2231
- **Detección Anomalías**: ~5% anomalías detectadas ( Isolation Forest)

---

## ESTRUCTURA DEL PROYECTO

```
smart_system_metro_medellin/
├── help.py                          # Ayuda general (todos los comandos)
├── main.py                           # Sistema de busqueda Dijkstra
├── main_ml.py                        # Aprendizaje supervisado
├── main_unsupervised.py               # Aprendizaje no supervisado
├── comando_ml.py                     # Comandos ML supervisado
├── comando_unsupervised.py           # Comandos ML no supervisado
├── README.md
│
├── ai/
│   └── sistema_inteligente.py        # Reglas logicas
│
├── core/
│   ├── motor_busqueda.py             # Algoritmo Dijkstra
│   └── sistema_transporte.py        # Grafo del sistema
│
├── data/
│   ├── setup.py                      # Configuracion del grafo
│   ├── dataset_generator.py          # Generador datasets supervisados
│   ├── dataset_generator_unsupervised.py # Generador datasets no supervisados
│   ├── dataset_*.csv                 # Datasets generados
│
└── models/
    ├── conexion.py                   # Modelo conexiones
    ├── prediccion_tiempo.py          # Regresion tiempo
    ├── prediccion_demanda.py         # Regresion demanda
    ├── clasificador_transporte.py    # Clasificacion
    ├── clustering_model.py           # Clustering
    ├── deteccion_anomalias.py       # Deteccion anomalias
    ├── reduccion_dimensional.py      # PCA y t-SNE
    └── *.pkl                         # Modelos guardados
```

---

## ALGORITMOS UTILIZADOS

### Aprendizaje Supervisado
- **Random Forest Regressor**: Predicción de valores continuos
- **Random Forest Classifier**: Clasificación multiclase

### Aprendizaje No Supervisado
- **K-Means**: Agrupamiento (clustering)
- **Isolation Forest**: Detección de anomalías
- **PCA**: Reducción dimensional lineal
- **t-SNE**: Reducción dimensional no lineal

### Sistema de Búsqueda
- **Dijkstra**: Camino más corto ponderado por tiempo

---

## MÉTRICAS UTILIZADAS

### Regresión
- MAE (Error Absoluto Medio)
- RMSE (Raíz del Error Cuadrático Medio)
- R² (Coeficiente de Determinación)

### Clasificación
- Accuracy (Exactitud)
- Precision / Recall / F1-Score
- Matriz de Confusión

### Clustering
- Silhouette Score
- Calinski-Harabasz Index
- Davies-Bouldin Index

---

## Cómo Extender el Proyecto

1. **Agregar más estaciones**: Editar `data/setup.py`
2. **Nuevos tipos de transporte**: Agregar en `data/setup.py`
3. **Mejorar reglas inteligentes**: Editar `ai/sistema_inteligente.py`
4. **Integrar algoritmos avanzados**: Reemplazar Dijkstra con A*
5. **Interfaz gráfica**: Crear con Streamlit o Flask

---

Este proyecto es educativo y puede expandirse para aplicaciones reales en planificación urbana o apps de transporte.

¡Disfruta explorando el transporte de Medellín de manera inteligente!
