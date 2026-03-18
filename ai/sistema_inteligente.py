from core.motor_busqueda import MotorBusqueda

class SistemaInteligente:
    def __init__(self, sistema):
        self.sistema = sistema
        self.motor = MotorBusqueda(sistema)

    # Contar transbordos correctamente
    def analizar_ruta(self, ruta):
        transbordos = 0
        ultimo_tipo = None

        for i in range(len(ruta) - 1):
            origen = ruta[i]
            destino = ruta[i + 1]

            for conexion in self.sistema.obtener_vecinos(origen):
                if conexion.destino == destino:
                    if ultimo_tipo is not None and conexion.tipo != ultimo_tipo and conexion.tipo != "Caminando":
                        transbordos += 1
                    ultimo_tipo = conexion.tipo
                    break

        return transbordos

    # Mostrar detalle de transporte
    def mostrar_ruta_detallada(self, ruta):
        print("\nDetalle de la ruta:")

        for i in range(len(ruta) - 1):
            origen = ruta[i]
            destino = ruta[i + 1]

            for conexion in self.sistema.obtener_vecinos(origen):
                if conexion.destino == destino:
                    print(f"{origen} -> {destino} [{conexion.tipo}]")
                    break

    # Calcular costo REAL basado en reglas
    def calcular_costo_real(self, ruta):
        tipos_usados = set()

        for i in range(len(ruta) - 1):
            origen = ruta[i]
            destino = ruta[i + 1]

            for conexion in self.sistema.obtener_vecinos(origen):
                if conexion.destino == destino:
                    tipos_usados.add(conexion.tipo)
                    break

        costo = 0

        # reglas del sistema real
        if "Metro" in tipos_usados:
            costo += 3200

        if "Bus" in tipos_usados:
            costo += 2800

        if "Taxi" in tipos_usados:
            costo += 10000

        return costo

    def ejecutar(self):
        print("=== Sistema Inteligente Transporte Valle de Aburrá ===")

        origen = input("Origen: ")
        destino = input("Destino: ")

        ruta, tiempo = self.motor.mejor_ruta(origen, destino)

        if ruta:
            print("\nRuta encontrada:")
            print(" -> ".join(ruta))

            self.mostrar_ruta_detallada(ruta)

            print("\nTiempo total:", tiempo, "min")
            print("Transbordos:", self.analizar_ruta(ruta))
            print("Costo real:", self.calcular_costo_real(ruta))
        else:
            print("No se encontró ruta")