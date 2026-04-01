import random
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class GeneradorDataset:
    def __init__(self, seed=42):
        random.seed(seed)
        np.random.seed(seed)
        
        self.estaciones = [
            "Niquia", "Bello", "Madera", "Acevedo", "Tricentenario",
            "Caribe", "Universidad", "Hospital", "Prado",
            "Parque Berrio", "San Antonio", "Alpujarra",
            "Exposiciones", "Industriales", "Poblado",
            "Aguacatala", "Ayura", "Envigado", "Itagui",
            "Sabaneta", "La Estrella", "San Javier", "Santa Lucia",
            "Floresta", "Estadio", "Suramericana", "Cisneros"
        ]
        
        self.tipos_transporte = ["Metro", "Bus", "Taxi", "Cable", "Caminando"]
        
        self.dias_semana = ["Lunes", "Martes", "Miercoles", "Jueves", "Viernes", "Sabado", "Domingo"]
        
        self.tiempos_base = {
            "Metro": 3, "Bus": 5, "Taxi": 15, "Cable": 10, "Caminando": 10
        }
        
        self.costos_base = {
            "Metro": 3200, "Bus": 2800, "Taxi": 12000, "Cable": 0, "Caminando": 0
        }
        
        self.picos = {
            "Mañana": (6, 8),
            "Mediodia": (11, 14),
            "Tarde": (17, 19),
            "Noche": (20, 22)
        }

    def generar_fechas(self, n_muestras, fecha_inicio=None, fecha_fin=None):
        if fecha_inicio is None:
            fecha_inicio = datetime(2025, 1, 1)
        if fecha_fin is None:
            fecha_fin = datetime(2025, 12, 31)
        
        fechas = []
        delta = fecha_fin - fecha_inicio
        
        for _ in range(n_muestras):
            dias_random = random.randint(0, delta.days)
            hora_random = random.randint(0, 23)
            minuto_random = random.choice([0, 15, 30, 45])
            fecha = fecha_inicio + timedelta(days=dias_random, hours=hora_random, minutes=minuto_random)
            fechas.append(fecha)
        
        return fechas

    def obtener_dia_semana(self, fecha):
        dias = ["Lunes", "Martes", "Miercoles", "Jueves", "Viernes", "Sabado", "Domingo"]
        return dias[fecha.weekday()]

    def es_hora_pico(self, hora):
        if 6 <= hora <= 8:
            return 1
        elif 11 <= hora <= 14:
            return 1
        elif 17 <= hora <= 19:
            return 1
        return 0

    def es_finde_semana(self, dia_semana):
        return 1 if dia_semana in ["Sabado", "Domingo"] else 0

    def calcular_tiempo_viaje(self, tipo_transporte, hora, dia_semana, distancia_km):
        tiempo_base = self.tiempos_base.get(tipo_transporte, 5)
        
        factor_pico = 1.5 if self.es_hora_pico(hora) else 1.0
        factor_findesemana = 0.7 if self.es_finde_semana(dia_semana) else 1.0
        
        variacion_aleatoria = random.uniform(0.8, 1.3)
        
        tiempo = (tiempo_base * distancia_km / 3) * factor_pico * factor_findesemana * variacion_aleatoria
        
        if random.random() < 0.05:
            tiempo *= random.uniform(1.5, 2.5)
        
        return max(1, round(tiempo, 1))

    def calcular_costo(self, tipo_transporte, distancia_km, hora):
        costo_base = self.costos_base.get(tipo_transporte, 2000)
        
        if tipo_transporte == "Taxi":
            costo_base = distancia_km * 800 + 3500
        
        factor_pico = 1.2 if self.es_hora_pico(hora) else 1.0
        
        return round(costo_base * factor_pico)

    def generar_afluencia_estacion(self, estacion, hora, dia_semana):
        estaciones_importantes = ["San Antonio", "Alpujarra", "Acevedo", "Poblado", "Niquia"]
        
        base_afluencia = random.randint(500, 2000)
        
        if estacion in estaciones_importantes:
            base_afluencia *= 2
        
        if self.es_hora_pico(hora):
            base_afluencia *= 1.8
        elif hora < 6 or hora > 22:
            base_afluencia *= 0.3
        
        if self.es_finde_semana(dia_semana):
            if estacion in ["Poblado", "Parque Berrio", "Exposiciones"]:
                base_afluencia *= 1.5
            else:
                base_afluencia *= 0.6
        
        return round(base_afluencia * random.uniform(0.7, 1.3))

    def generar_dataset_viajes(self, n_muestras=5000):
        datos = []
        
        for _ in range(n_muestras):
            origen = random.choice(self.estaciones)
            destino = random.choice([e for e in self.estaciones if e != origen])
            tipo_transporte = random.choice(self.tipos_transporte)
            
            fecha = self.generar_fechas(1)[0]
            dia_semana = self.obtener_dia_semana(fecha)
            hora = fecha.hour
            
            distancia_km = round(random.uniform(1, 15), 1)
            
            tiempo_viaje = self.calcular_tiempo_viaje(tipo_transporte, hora, dia_semana, distancia_km)
            costo = self.calcular_costo(tipo_transporte, distancia_km, hora)
            
            transferencia = 1 if random.random() < 0.3 else 0
            
            dato = {
                "origen": origen,
                "destino": destino,
                "tipo_transporte": tipo_transporte,
                "distancia_km": distancia_km,
                "dia_semana": dia_semana,
                "hora": hora,
                "es_hora_pico": self.es_hora_pico(hora),
                "es_finde_semana": self.es_finde_semana(dia_semana),
                "tiempo_viaje_min": tiempo_viaje,
                "costo": costo,
                "transferencias": transferencia,
                "fecha": fecha.strftime("%Y-%m-%d"),
                "clima": random.choice(["Soleado", "Nublado", "Lluvia", "Tormenta"])
            }
            datos.append(dato)
        
        return pd.DataFrame(datos)

    def generar_dataset_afluencia(self, n_muestras=3000):
        datos = []
        
        for _ in range(n_muestras):
            estacion = random.choice(self.estaciones)
            
            fecha = self.generar_fechas(1)[0]
            dia_semana = self.obtener_dia_semana(fecha)
            hora = fecha.hour
            
            afluencias = []
            for _ in range(4):
                afluencias.append(self.generar_afluencia_estacion(estacion, hora, dia_semana))
            
            capacidad_max = 5000
            ocupacion_promedio = sum(afluencias) / len(afluencias)
            porcentaje_ocupacion = (ocupacion_promedio / capacidad_max) * 100
            
            es_hora_pico = self.es_hora_pico(hora)
            es_finde = self.es_finde_semana(dia_semana)
            
            espera_min = round(random.uniform(2, 15) * (1.5 if es_hora_pico else 1), 1)
            
            if porcentaje_ocupacion > 80:
                nivel_congestion = "Alto"
            elif porcentaje_ocupacion > 50:
                nivel_congestion = "Medio"
            else:
                nivel_congestion = "Bajo"
            
            dato = {
                "estacion": estacion,
                "dia_semana": dia_semana,
                "hora": hora,
                "es_hora_pico": es_hora_pico,
                "es_finde_semana": es_finde,
                "afluencia_1": afluencias[0],
                "afluencia_2": afluencias[1],
                "afluencia_3": afluencias[2],
                "afluencia_4": afluencias[3],
                "afluencia_promedio": round(ocupacion_promedio, 1),
                "capacidad_estacion": capacidad_max,
                "porcentaje_ocupacion": round(porcentaje_ocupacion, 1),
                "tiempo_espera_min": espera_min,
                "nivel_congestion": nivel_congestion,
                "clima": random.choice(["Soleado", "Nublado", "Lluvia", "Tormenta"]),
                "evento_especial": random.choice([0, 0, 0, 1])
            }
            datos.append(dato)
        
        return pd.DataFrame(datos)

    def generar_dataset_transporte(self, n_muestras=4000):
        datos = []
        
        for _ in range(n_muestras):
            origen = random.choice(self.estaciones)
            destino = random.choice([e for e in self.estaciones if e != origen])
            
            fecha = self.generar_fechas(1)[0]
            dia_semana = self.obtener_dia_semana(fecha)
            hora = fecha.hour
            
            distancia_km = round(random.uniform(1, 20), 1)
            
            es_urgente = random.choice([0, 1])
            presupuesto = random.choice(["bajo", "medio", "alto"])
            
            if es_urgente == 1:
                transporte_optimo = "Taxi"
            elif presupuesto == "bajo" and distancia_km > 5:
                transporte_optimo = "Bus"
            elif distancia_km <= 2:
                transporte_optimo = random.choice(["Caminando", "Bus"])
            elif hora in [6, 7, 8, 17, 18, 19]:
                if distancia_km > 8:
                    transporte_optimo = random.choice(["Metro", "Bus"])
                else:
                    transporte_optimo = "Bus"
            else:
                transporte_optimo = random.choice(["Metro", "Bus", "Taxi"])
            
            dato = {
                "origen": origen,
                "destino": destino,
                "distancia_km": distancia_km,
                "dia_semana": dia_semana,
                "hora": hora,
                "es_hora_pico": self.es_hora_pico(hora),
                "es_finde_semana": self.es_finde_semana(dia_semana),
                "es_urgente": es_urgente,
                "presupuesto": presupuesto,
                "transporte_optimo": transporte_optimo
            }
            datos.append(dato)
        
        return pd.DataFrame(datos)

    def guardar_datasets(self, directorio="data"):
        import os
        os.makedirs(directorio, exist_ok=True)
        
        df_viajes = self.generar_dataset_viajes()
        df_viajes.to_csv(f"{directorio}/dataset_viajes.csv", index=False)
        print(f"Dataset de viajes guardado: {len(df_viajes)} registros")
        
        df_afluencia = self.generar_dataset_afluencia()
        df_afluencia.to_csv(f"{directorio}/dataset_afluencia.csv", index=False)
        print(f"Dataset de afluencia guardado: {len(df_afluencia)} registros")
        
        df_transporte = self.generar_dataset_transporte()
        df_transporte.to_csv(f"{directorio}/dataset_transporte.csv", index=False)
        print(f"Dataset de transporte guardado: {len(df_transporte)} registros")
        
        return df_viajes, df_afluencia, df_transporte


if __name__ == "__main__":
    generador = GeneradorDataset()
    generador.guardar_datasets()
    print("Todos los datasets han sido generados exitosamente.")
