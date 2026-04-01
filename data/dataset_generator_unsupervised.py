import random
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class GeneradorDatasetNoSupervisado:
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
        self.climas = ["Soleado", "Nublado", "Lluvia", "Tormenta"]
        self.dias_semana = ["Lunes", "Martes", "Miercoles", "Jueves", "Viernes", "Sabado", "Domingo"]

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

    def es_hora_pico(self, hora):
        return 1 if (6 <= hora <= 8 or 11 <= hora <= 14 or 17 <= hora <= 19) else 0

    def es_finde_semana(self, dia_semana):
        return 1 if dia_semana in ["Sabado", "Domingo"] else 0

    def generar_dataset_estaciones(self, n_muestras=1000):
        datos = []
        
        estaciones_importantes = ["San Antonio", "Alpujarra", "Acevedo", "Poblado", "Niquia"]
        estaciones_terminales = ["Niquia", "La Estrella", "San Javier"]
        
        for _ in range(n_muestras):
            estacion = random.choice(self.estaciones)
            fecha = self.generar_fechas(1)[0]
            dia_semana = self.dias_semana[fecha.weekday()]
            hora = fecha.hour
            
            if estacion in estaciones_importantes:
                base_demanda = random.randint(1500, 4000)
            elif estacion in estaciones_terminales:
                base_demanda = random.randint(800, 2000)
            else:
                base_demanda = random.randint(300, 1200)
            
            if self.es_hora_pico(hora):
                base_demanda = int(base_demanda * random.uniform(1.5, 2.0))
            
            if self.es_finde_semana(dia_semana):
                if estacion in ["Poblado", "Parque Berrio"]:
                    base_demanda = int(base_demanda * random.uniform(1.3, 1.8))
                else:
                    base_demanda = int(base_demanda * random.uniform(0.5, 0.8))
            
            tiempo_espera = random.uniform(2, 15) * (1.3 if self.es_hora_pico(hora) else 0.9)
            
            transferencias_disponibles = random.randint(0, 4)
            
            ingresos = base_demanda * random.uniform(2800, 3500)
            
            if random.random() < 0.05:
                tiempo_espera *= random.uniform(2.0, 3.0)
                base_demanda = int(base_demanda * 0.3)
            
            tipos_disponibles = random.sample(self.tipos_transporte, k=random.randint(1, 4))
            
            dato = {
                "estacion": estacion,
                "dia_semana": dia_semana,
                "hora": hora,
                "es_hora_pico": self.es_hora_pico(hora),
                "es_finde_semana": self.es_finde_semana(dia_semana),
                "demanda_pasajeros": base_demanda,
                "tiempo_espera_min": round(tiempo_espera, 1),
                "transferencias_disponibles": transferencias_disponibles,
                "ingresos_diarios": round(ingresos, 0),
                "clima": random.choice(self.climas),
                "num_tipos_transporte": len(tipos_disponibles),
                "tiene_metro": 1 if "Metro" in tipos_disponibles else 0,
                "tiene_bus": 1 if "Bus" in tipos_disponibles else 0,
                "tiene_taxi": 1 if "Taxi" in tipos_disponibles else 0,
                "tiene_cable": 1 if "Cable" in tipos_disponibles else 0,
                "evento_especial": random.choice([0, 0, 0, 1])
            }
            datos.append(dato)
        
        return pd.DataFrame(datos)

    def generar_dataset_viajes_sin_etiquetas(self, n_muestras=2000):
        datos = []
        
        for _ in range(n_muestras):
            origen = random.choice(self.estaciones)
            destino = random.choice([e for e in self.estaciones if e != origen])
            tipo_transporte = random.choice(self.tipos_transporte)
            
            fecha = self.generar_fechas(1)[0]
            dia_semana = self.dias_semana[fecha.weekday()]
            hora = fecha.hour
            
            distancia_km = round(random.uniform(1, 15), 1)
            
            if tipo_transporte == "Metro":
                tiempo = distancia_km / 3 * random.uniform(0.8, 1.2)
            elif tipo_transporte == "Bus":
                tiempo = distancia_km / 2.5 * random.uniform(0.7, 1.3)
            elif tipo_transporte == "Taxi":
                tiempo = distancia_km / 4 * random.uniform(0.6, 1.0)
            elif tipo_transporte == "Cable":
                tiempo = distancia_km / 5 * random.uniform(0.9, 1.1)
            else:
                tiempo = distancia_km / 0.5 * random.uniform(0.8, 1.2)
            
            if self.es_hora_pico(hora):
                tiempo *= random.uniform(1.3, 1.8)
            
            if random.random() < 0.03:
                tiempo *= random.uniform(2.0, 3.5)
            
            costo = distancia_km * random.randint(800, 1200)
            
            dato = {
                "origen": origen,
                "destino": destino,
                "distancia_km": distancia_km,
                "tipo_transporte": tipo_transporte,
                "dia_semana": dia_semana,
                "hora": hora,
                "es_hora_pico": self.es_hora_pico(hora),
                "es_finde_semana": self.es_finde_semana(dia_semana),
                "tiempo_viaje_min": round(tiempo, 1),
                "costo": round(costo),
                "transferencias": random.randint(0, 3),
                "clima": random.choice(self.climas),
                "ocupacion_estimada": random.randint(10, 100)
            }
            datos.append(dato)
        
        return pd.DataFrame(datos)

    def generar_dataset_perfiles_usuario(self, n_muestras=1500):
        datos = []
        
        perfiles = ["Ocasional", "Frecuente", "Premium", "Turista"]
        
        for _ in range(n_muestras):
            perfil = random.choice(perfiles)
            
            if perfil == "Ocasional":
                viajes_mes = random.randint(1, 8)
                presupuesto = "bajo"
                pref_horario = random.choice(["Mañana", "Tarde", "Noche"])
                usa_metro = random.choice([0, 1])
                usa_taxi = 0
            elif perfil == "Frecuente":
                viajes_mes = random.randint(20, 60)
                presupuesto = "bajo"
                pref_horario = random.choice(["Hora Pico", "Fuera de Hora Pico"])
                usa_metro = 1
                usa_taxi = 0
            elif perfil == "Premium":
                viajes_mes = random.randint(5, 25)
                presupuesto = "alto"
                pref_horario = random.choice(["Flexible"])
                usa_metro = random.choice([0, 1])
                usa_taxi = 1
            else:
                viajes_mes = random.randint(3, 15)
                presupuesto = "medio"
                pref_horario = "Tarde"
                usa_metro = 1
                usa_taxi = 0
            
            origen = random.choice(self.estaciones)
            destino = random.choice([e for e in self.estaciones if e != origen])
            
            dato = {
                "perfil": perfil,
                "viajes_mes": viajes_mes,
                "presupuesto": presupuesto,
                "pref_horario": pref_horario,
                "origen_frecuente": origen,
                "destino_frecuente": destino,
                "usa_metro": usa_metro,
                "usa_taxi": usa_taxi,
                "distancia_promedio": round(random.uniform(3, 15), 1),
                "tolerancia_transbordos": random.randint(0, 3),
                "usa_horario_pico": 1 if random.random() > 0.3 else 0,
                "eventos_mes": random.randint(0, 8)
            }
            datos.append(dato)
        
        return pd.DataFrame(datos)

    def guardar_datasets(self, directorio="data"):
        import os
        os.makedirs(directorio, exist_ok=True)
        
        df_estaciones = self.generar_dataset_estaciones()
        df_estaciones.to_csv(f"{directorio}/dataset_clustering_estaciones.csv", index=False)
        print(f"Dataset clustering estaciones guardado: {len(df_estaciones)} registros")
        
        df_viajes = self.generar_dataset_viajes_sin_etiquetas()
        df_viajes.to_csv(f"{directorio}/dataset_clustering_viajes.csv", index=False)
        print(f"Dataset clustering viajes guardado: {len(df_viajes)} registros")
        
        df_perfiles = self.generar_dataset_perfiles_usuario()
        df_perfiles.to_csv(f"{directorio}/dataset_perfiles_usuario.csv", index=False)
        print(f"Dataset perfiles guardado: {len(df_perfiles)} registros")
        
        return df_estaciones, df_viajes, df_perfiles


if __name__ == "__main__":
    generador = GeneradorDatasetNoSupervisado()
    generador.guardar_datasets()
    print("Todos los datasets de aprendizaje no supervisado han sido generados.")
