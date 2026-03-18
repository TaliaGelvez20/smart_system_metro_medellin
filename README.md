# Sistema Inteligente de Transporte Valle de Aburrá

Este proyecto implementa un **sistema inteligente de transporte** para el Valle de Aburrá (Medellín, Colombia), diseñado como una actividad académica en inteligencia artificial. Utiliza un enfoque basado en **búsqueda y sistemas de reglas lógicas** para encontrar la mejor ruta entre dos puntos en el sistema de transporte masivo, considerando tiempo, costo económico y eficiencia.

## Funcionalidad

El sistema modela la red de transporte como un **grafo multimodal**, donde:
- **Nodos**: Estaciones y puntos de interés (ej. estaciones de Metro, terminales de buses, centros comerciales).
- **Aristas**: Conexiones entre nodos con atributos como tipo de transporte (Metro, Bus, Taxi, Cable, Caminando), tiempo de viaje (minutos) y costo monetario (pesos colombianos).

### Componentes Principales
- **SistemaTransporte** (`core/sistema_transporte.py`): Gestiona el grafo, agregando estaciones y conexiones.
- **MotorBusqueda** (`core/motor_busqueda.py`): Implementa el algoritmo de Dijkstra para encontrar la ruta más corta en términos de tiempo.
- **SistemaInteligente** (`ai/sistema_inteligente.py`): Aplica reglas lógicas inteligentes, como:
  - Penalización por transbordos (cambios de tipo de transporte).
  - Preferencia por Metro (más eficiente).
  - Desincentivo al uso de taxis (costos altos).
  - Cálculo de costo real basado en tarifas por tipo de transporte usado.
- **Setup** (`data/setup.py`): Configura el grafo con datos reales del Metro de Medellín (Líneas A y B), buses integrados y conexiones adicionales.

### Algoritmo
- **Base**: Dijkstra para caminos más cortos (optimizado por tiempo).
- **Inteligencia**: Reglas lógicas aplicadas post-búsqueda para evaluar eficiencia y costos reales, simulando un sistema basado en conocimiento.

## Requisitos
- Python 3.7 o superior.
- No se requieren bibliotecas externas (usa solo módulos estándar como `heapq` y `collections`).

## Cómo Ejecutar el Proyecto

1. **Clona o descarga el repositorio** en tu máquina local.
2. **Navega al directorio del proyecto**:
   ```
   cd smart_system_metro_medellin
   ```
3. **Ejecuta el programa principal**:
   ```
   python main.py
   ```
4. **Interacción**:
   - Ingresa el **origen** (ej. "Niquia").
   - Ingresa el **destino** (ej. "Poblado").
   - El sistema mostrará:
     - La ruta óptima.
     - Detalle de cada tramo (tipo de transporte).
     - Tiempo total (minutos).
     - Número de transbordos.
     - Costo real estimado (basado en reglas).

### Ejemplo de Salida
```
=== Sistema Inteligente Transporte Valle de Aburrá ===

Origen: Niquia
Destino: Poblado

Ruta encontrada:
Niquia -> Bello -> Madera -> Acevedo -> Tricentenario -> Caribe -> Universidad -> Hospital -> Prado -> Parque Berrio -> San Antonio -> Alpujarra -> Exposiciones -> Industriales -> Poblado

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

El proyecto está diseñado para ser modular y extensible. Aquí van algunas formas de agregarle funcionalidades:

### 1. **Agregar Más Estaciones y Conexiones**
   - Edita `data/setup.py`:
     - Agrega nuevas estaciones llamando a `sistema.agregar_estacion("NuevaEstacion")`.
     - Conecta estaciones con `sistema.conectar(origen, destino, tipo, tiempo, costo)`.
     - Ejemplo: Agregar una nueva línea de Metro o buses a municipios adicionales.
   - Actualiza tiempos y costos con datos reales para mayor precisión.

### 2. **Agregar Nuevos Tipos de Transporte**
   - En `data/setup.py`, usa tipos como "Bicicleta", "Cable" o "Tranvía".
   - Actualiza las reglas en `ai/sistema_inteligente.py` para incluir costos o penalizaciones para nuevos tipos (ej. agregar al método `calcular_costo_real`).

### 3. **Mejorar las Reglas Inteligentes**
   - Edita `ai/sistema_inteligente.py`:
     - Modifica `analizar_ruta` para contar transbordos de manera más sofisticada (ej. penalizar ciertos cambios).
     - Ajusta `calcular_costo_real` para incluir descuentos (ej. por hora pico) o factores ambientales (ej. preferir transporte ecológico).
     - Agrega un archivo `knowledge_base.json` para reglas externas (ej. reglas en formato JSON para cargar dinámicamente).

### 4. **Integrar Algoritmos Avanzados**
   - Reemplaza Dijkstra en `core/motor_busqueda.py` con A* (agregando heurísticas, como distancia euclidiana entre estaciones).
   - Agrega aprendizaje automático (ej. usar datos históricos para predecir tiempos en hora pico).

### 5. **Interfaz Gráfica o Web**
   - Crea una interfaz con Tkinter, Flask o Streamlit para visualizar el grafo y rutas.
   - Agrega mapas interactivos usando bibliotecas como Folium.

### 6. **Validación y Pruebas**
   - Agrega pruebas unitarias con `unittest` para verificar rutas y costos.
   - Valida con datos reales del Metro de Medellín.

### Estructura del Proyecto
```
smart_system_metro_medellin/
├── main.py                 # Punto de entrada
├── README.md               # Este archivo
├── ai/
│   └── sistema_inteligente.py  # Lógica inteligente y reglas
├── core/
│   ├── motor_busqueda.py   # Algoritmo de búsqueda (Dijkstra)
│   └── sistema_transporte.py  # Gestión del grafo
├── data/
│   └── setup.py            # Configuración del grafo
└── models/
    └── conexion.py         # Modelo de conexiones
```

## Contribución
Si deseas contribuir:
1. Haz un fork del repositorio.
2. Crea una rama para tu feature.
3. Envía un pull request con cambios bien documentados.

Este proyecto es educativo y puede expandirse para aplicaciones reales en planificación urbana o apps de transporte.

¡Disfruta explorando el transporte de Medellín de manera inteligente!