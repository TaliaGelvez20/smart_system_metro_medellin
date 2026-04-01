import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import os

class ModeloDeteccionAnomalias:
    def __init__(self):
        self.modelo = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.caracteristicas = None
        self.algoritmo = None
        self.esta_entrenado = False
        
    def preprocesar_datos(self, df, tipo="estaciones"):
        df_procesado = df.copy()
        
        if tipo == "estaciones":
            le_estacion = LabelEncoder()
            le_dia = LabelEncoder()
            le_clima = LabelEncoder()
            
            df_procesado["estacion_enc"] = le_estacion.fit_transform(df_procesado["estacion"])
            df_procesado["dia_semana_enc"] = le_dia.fit_transform(df_procesado["dia_semana"])
            df_procesado["clima_enc"] = le_clima.fit_transform(df_procesado["clima"])
            
            self.label_encoders = {
                "estacion": le_estacion,
                "dia_semana": le_dia,
                "clima": le_clima
            }
            
            self.caracteristicas = [
                "demanda_pasajeros", "tiempo_espera_min", "transferencias_disponibles",
                "ingresos_diarios", "es_hora_pico", "es_finde_semana"
            ]
            
        elif tipo == "viajes":
            le_origen = LabelEncoder()
            le_destino = LabelEncoder()
            le_tipo = LabelEncoder()
            le_dia = LabelEncoder()
            
            df_procesado["origen_enc"] = le_origen.fit_transform(df_procesado["origen"])
            df_procesado["destino_enc"] = le_destino.fit_transform(df_procesado["destino"])
            df_procesado["tipo_transporte_enc"] = le_tipo.fit_transform(df_procesado["tipo_transporte"])
            df_procesado["dia_semana_enc"] = le_dia.fit_transform(df_procesado["dia_semana"])
            
            self.label_encoders = {
                "origen": le_origen,
                "destino": le_destino,
                "tipo_transporte": le_tipo,
                "dia_semana": le_dia
            }
            
            self.caracteristicas = [
                "distancia_km", "tiempo_viaje_min", "costo",
                "transferencias", "es_hora_pico", "es_finde_semana", "ocupacion_estimada"
            ]
        
        X = df_procesado[self.caracteristicas]
        
        return X
    
    def entrenar(self, df, tipo="estaciones", algoritmo="isolation_forest", contaminacion=0.05):
        print("=" * 60)
        print(f"ENTRENAMIENTO: Deteccion de Anomalias ({tipo.upper()})")
        print("=" * 60)
        
        X = self.preprocesar_datos(df, tipo)
        
        X_scaled = self.scaler.fit_transform(X)
        
        print(f"\nDatos: {len(X)} muestras")
        print(f"Caracteristicas: {self.caracteristicas}")
        print(f"Algoritmo: {algoritmo}")
        print(f"Contaminacion esperada: {contaminacion*100:.1f}%")
        
        self.algoritmo = algoritmo
        
        if algoritmo == "isolation_forest":
            self.modelo = IsolationForest(
                n_estimators=100,
                contamination=contaminacion,
                random_state=42,
                n_jobs=-1
            )
            labels = self.modelo.fit_predict(X_scaled)
            scores = self.modelo.score_samples(X_scaled)
            
        elif algoritmo == "lof":
            self.modelo = LocalOutlierFactor(
                n_neighbors=20,
                contamination=contaminacion,
                n_jobs=-1
            )
            labels = self.modelo.fit_predict(X_scaled)
            scores = self.modelo.negative_outlier_factor_
            
        elif algoritmo == "one_class_svm":
            self.modelo = OneClassSVM(
                kernel='rbf',
                gamma='auto',
                nu=contaminacion
            )
            labels = self.modelo.fit_predict(X_scaled)
            scores = self.modelo.decision_function(X_scaled)
            
        else:
            raise ValueError(f"Algoritmo no reconocido: {algoritmo}")
        
        df_resultado = df.copy()
        df_resultado["es_anomalia"] = (labels == -1).astype(int)
        df_resultado["score_anomalia"] = scores
        
        n_anomalias = (labels == -1).sum()
        n_normales = (labels == 1).sum()
        
        print("\n" + "-" * 50)
        print("RESULTADOS DETECCION DE ANOMALIAS")
        print("-" * 50)
        print(f"  Total muestras: {len(df_resultado)}")
        print(f"  Normales: {n_normales} ({n_normales/len(df_resultado)*100:.1f}%)")
        print(f"  Anomalias detectadas: {n_anomalias} ({n_anomalias/len(df_resultado)*100:.1f}%)")
        
        print("\n" + "-" * 50)
        print("ANOMALIAS DETECTADAS (primeras 10)")
        print("-" * 50)
        
        anomalias = df_resultado[df_resultado["es_anomalia"] == 1].head(10)
        
        if tipo == "estaciones":
            for _, row in anomalias.iterrows():
                print(f"  Estacion: {row['estacion']}, Hora: {row['hora']}:00")
                print(f"    Demanda: {row['demanda_pasajeros']}, Tiempo espera: {row['tiempo_espera_min']} min")
                print(f"    Score: {row['score_anomalia']:.4f}")
                
        elif tipo == "viajes":
            for _, row in anomalias.iterrows():
                print(f"  Viaje: {row['origen']} -> {row['destino']}")
                print(f"    Tipo: {row['tipo_transporte']}, Tiempo: {row['tiempo_viaje_min']} min")
                print(f"    Score: {row['score_anomalia']:.4f}")
        
        print("\n" + "-" * 50)
        print("ESTADISTICAS DE SCORES")
        print("-" * 50)
        print(f"  Score normal (media): {df_resultado[df_resultado['es_anomalia']==0]['score_anomalia'].mean():.4f}")
        print(f"  Score anomalia (media): {df_resultado[df_resultado['es_anomalia']==1]['score_anomalia'].mean():.4f}")
        
        self.esta_entrenado = True
        
        return df_resultado
    
    def predecir(self, datos):
        if not self.esta_entrenado:
            raise ValueError("El modelo debe ser entrenado primero")
        
        df = pd.DataFrame([datos])
        
        for col in self.caracteristicas:
            if col not in df.columns:
                df[col] = 0
        
        X = df[self.caracteristicas]
        X_scaled = self.scaler.transform(X)
        
        label = self.modelo.predict(X_scaled)[0]
        score = self.modelo.score_samples(X_scaled)[0]
        
        es_anomalia = label == -1
        
        return {
            "es_anomalia": bool(es_anomalia),
            "score_anomalia": float(score),
            "tipo": "ANOMALIA" if es_anomalia else "NORMAL"
        }
    
    def detectar_en_datos(self, df_nuevos):
        X = df_nuevos[self.caracteristicas]
        X_scaled = self.scaler.transform(X)
        
        labels = self.modelo.predict(X_scaled)
        scores = self.modelo.score_samples(X_scaled)
        
        df_resultado = df_nuevos.copy()
        df_resultado["es_anomalia"] = (labels == -1).astype(int)
        df_resultado["score_anomalia"] = scores
        
        return df_resultado
    
    def guardar_modelo(self, ruta="models/deteccion_anomalias_model.pkl"):
        os.makedirs(os.path.dirname(ruta), exist_ok=True)
        joblib.dump({
            "modelo": self.modelo,
            "scaler": self.scaler,
            "label_encoders": self.label_encoders,
            "caracteristicas": self.caracteristicas,
            "algoritmo": self.algoritmo
        }, ruta)
        print(f"\nModelo guardado en: {ruta}")
    
    def cargar_modelo(self, ruta="models/deteccion_anomalias_model.pkl"):
        datos = joblib.load(ruta)
        self.modelo = datos["modelo"]
        self.scaler = datos["scaler"]
        self.label_encoders = datos["label_encoders"]
        self.caracteristicas = datos["caracteristicas"]
        self.algoritmo = datos["algoritmo"]
        self.esta_entrenado = True
        print(f"Modelo cargado desde: {ruta}")


def main():
    from data.dataset_generator_unsupervised import GeneradorDatasetNoSupervisado
    
    print("\n" + "=" * 60)
    print("DETECCION DE ANOMALIAS - SISTEMA METRO MEDELLIN")
    print("=" * 60)
    
    generador = GeneradorDatasetNoSupervisado()
    
    print("\n1. DETECCION EN ESTACIONES")
    df_estaciones = generador.generar_dataset_estaciones(n_muestras=500)
    modelo_est = ModeloDeteccionAnomalias()
    resultado_est = modelo_est.entrenar(df_estaciones, tipo="estaciones", algoritmo="isolation_forest")
    modelo_est.guardar_modelo("models/deteccion_anomalias_estaciones.pkl")
    
    print("\n2. DETECCION EN VIAJES")
    df_viajes = generador.generar_dataset_viajes_sin_etiquetas(n_muestras=600)
    modelo_viajes = ModeloDeteccionAnomalias()
    resultado_viajes = modelo_viajes.entrenar(df_viajes, tipo="viajes", algoritmo="isolation_forest")
    modelo_viajes.guardar_modelo("models/deteccion_anomalias_viajes.pkl")
    
    print("\n3. COMPARACION DE ALGORITMOS")
    print("-" * 50)
    
    algoritmos = ["isolation_forest", "lof", "one_class_svm"]
    for alg in algoritmos:
        modelo = ModeloDeteccionAnomalias()
        resultado = modelo.entrenar(df_estaciones.head(300), tipo="estaciones", algoritmo=alg)
        n_anom = resultado["es_anomalia"].sum()
        print(f"  {alg}: {n_anom} anomalias detectadas")


if __name__ == "__main__":
    main()
