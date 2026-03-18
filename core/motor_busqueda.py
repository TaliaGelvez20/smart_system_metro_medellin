import heapq

class MotorBusqueda:
    def __init__(self, sistema):
        self.sistema = sistema

    def mejor_ruta(self, origen, destino):
        cola = [(0, origen, [])]
        visitados = set()

        while cola:
            costo, actual, ruta = heapq.heappop(cola)

            if actual in visitados:
                continue

            ruta = ruta + [actual]
            visitados.add(actual)

            if actual == destino:
                return ruta, costo

            for conexion in self.sistema.obtener_vecinos(actual):
                if conexion.destino not in visitados:
                    # usamos tiempo como métrica real
                    heapq.heappush(
                        cola,
                        (costo + conexion.tiempo, conexion.destino, ruta)
                    )

        return None, float("inf")