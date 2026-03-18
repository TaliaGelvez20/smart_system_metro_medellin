from collections import defaultdict
from models.conexion import Conexion

class SistemaTransporte:
    def __init__(self):
        self.grafo = defaultdict(list)

    def agregar_estacion(self, nombre):
        if nombre not in self.grafo:
            self.grafo[nombre] = []

    def conectar(self, origen, destino, tipo, tiempo, costo_dinero):
        # Crear estaciones automáticamente si no existen
        if origen not in self.grafo:
            self.agregar_estacion(origen)

        if destino not in self.grafo:
            self.agregar_estacion(destino)

        self.grafo[origen].append(Conexion(destino, tipo, tiempo, costo_dinero))
        self.grafo[destino].append(Conexion(origen, tipo, tiempo, costo_dinero))

    def obtener_vecinos(self, estacion):
        return self.grafo[estacion]