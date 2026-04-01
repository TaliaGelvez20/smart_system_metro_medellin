import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import joblib
import os

class ModeloClustering:
    def __init__(self):
        self.modelo = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.caracteristicas = None
        self.n_clusters = None
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
                "ingresos_diarios", "num_tipos_transporte", "es_hora_pico",
                "es_finde_semana", "tiene_metro", "tiene_bus", "tiene_taxi", "tiene_cable"
            ]
            
        elif tipo == "viajes":
            le_origen = LabelEncoder()
            le_destino = LabelEncoder()
            le_tipo = LabelEncoder()
            le_dia = LabelEncoder()
            le_clima = LabelEncoder()
            
            df_procesado["origen_enc"] = le_origen.fit_transform(df_procesado["origen"])
            df_procesado["destino_enc"] = le_destino.fit_transform(df_procesado["destino"])
            df_procesado["tipo_transporte_enc"] = le_tipo.fit_transform(df_procesado["tipo_transporte"])
            df_procesado["dia_semana_enc"] = le_dia.fit_transform(df_procesado["dia_semana"])
            df_procesado["clima_enc"] = le_clima.fit_transform(df_procesado["clima"])
            
            self.label_encoders = {
                "origen": le_origen,
                "destino": le_destino,
                "tipo_transporte": le_tipo,
                "dia_semana": le_dia,
                "clima": le_clima
            }
            
            self.caracteristicas = [
                "distancia_km", "tiempo_viaje_min", "costo",
                "transferencias", "es_hora_pico", "es_finde_semana",
                "ocupacion_estimada"
            ]
            
        elif tipo == "perfiles":
            le_perfil = LabelEncoder()
            le_presupuesto = LabelEncoder()
            le_horario = LabelEncoder()
            
            df_procesado["perfil_enc"] = le_perfil.fit_transform(df_procesado["perfil"])
            df_procesado["presupuesto_enc"] = le_presupuesto.fit_transform(df_procesado["presupuesto"])
            df_procesado["pref_horario_enc"] = le_horario.fit_transform(df_procesado["pref_horario"])
            
            self.label_encoders = {
                "perfil": le_perfil,
                "presupuesto": le_presupuesto,
                "pref_horario": le_horario
            }
            
            self.caracteristicas = [
                "viajes_mes", "distancia_promedio", "tolerancia_transbordos",
                "usa_horario_pico", "eventos_mes", "usa_metro", "usa_taxi"
            ]
        
        X = df_procesado[self.caracteristicas]
        
        return X
    
    def encontrar_k_optimo(self, X_scaled, max_k=10):
        print("\nBuscando numero optimo de clusters (K)...")
        
        inertias = []
        silhouettes = []
        k_range = range(2, min(max_k + 1, len(X_scaled)))
        
        for k in k_range:
            kmeans_temp = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels_temp = kmeans_temp.fit_predict(X_scaled)
            
            inertias.append(kmeans_temp.inertia_)
            silhouettes.append(silhouette_score(X_scaled, labels_temp))
        
        mejor_k = list(k_range)[np.argmax(silhouettes)]
        
        print(f"\nK optimo encontrado: {mejor_k}")
        print("\nMetricas por K:")
        print("-" * 50)
        print(f"{'K':<5} {'Inertia':<15} {'Silhouette':<15}")
        print("-" * 50)
        for i, k in enumerate(k_range):
            print(f"{k:<5} {inertias[i]:<15.2f} {silhouettes[i]:<15.4f}")
        
        return mejor_k
    
    def entrenar(self, df, n_clusters=None, tipo="estaciones", algoritmo="kmeans"):
        print("=" * 60)
        print(f"ENTRENAMIENTO: Clustering de {tipo.upper()}")
        print("=" * 60)
        
        X = self.preprocesar_datos(df, tipo)
        
        X_scaled = self.scaler.fit_transform(X)
        
        print(f"\nDatos: {len(X)} muestras, {len(self.caracteristicas)} caracteristicas")
        print(f"Caracteristicas: {self.caracteristicas}")
        
        if n_clusters is None:
            n_clusters = self.encontrar_k_optimo(X_scaled)
        
        self.n_clusters = n_clusters
        
        print(f"\nEntrenando {algoritmo} con {n_clusters} clusters...")
        
        if algoritmo == "kmeans":
            self.modelo = KMeans(
                n_clusters=n_clusters,
                random_state=42,
                n_init=10,
                max_iter=300
            )
        elif algoritmo == "hierarchical":
            self.modelo = AgglomerativeClustering(n_clusters=n_clusters)
        else:
            raise ValueError(f"Algoritmo no reconocido: {algoritmo}")
        
        if algoritmo == "kmeans":
            labels = self.modelo.fit_predict(X_scaled)
            self.modelo.fit(X_scaled)
        else:
            labels = self.modelo.fit_predict(X_scaled)
        
        print("\n" + "-" * 50)
        print("METRICAS DE CLUSTERING")
        print("-" * 50)
        print(f"  Silhouette Score: {silhouette_score(X_scaled, labels):.4f}")
        print(f"  Calinski-Harabasz: {calinski_harabasz_score(X_scaled, labels):.2f}")
        print(f"  Davies-Bouldin: {davies_bouldin_score(X_scaled, labels):.4f}")
        
        df_resultado = df.copy()
        df_resultado["cluster"] = labels
        
        print("\n" + "-" * 50)
        print("DISTRIBUCION DE CLUSTERS")
        print("-" * 50)
        distribucion = df_resultado["cluster"].value_counts().sort_index()
        for cluster_id, count in distribucion.items():
            print(f"  Cluster {cluster_id}: {count} muestras ({count/len(df_resultado)*100:.1f}%)")
        
        print("\n" + "-" * 50)
        print(f"PERFIL DE CADA CLUSTER (Promedio)")
        print("-" * 50)
        
        resumen_clusters = df_resultado.groupby("cluster")[self.caracteristicas].mean()
        print(resumen_clusters.round(2))
        
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
        
        cluster = self.modelo.predict(X_scaled)[0]
        
        return {"cluster_asignado": int(cluster)}
    
    def guardar_modelo(self, ruta="models/clustering_model.pkl"):
        os.makedirs(os.path.dirname(ruta), exist_ok=True)
        joblib.dump({
            "modelo": self.modelo,
            "scaler": self.scaler,
            "label_encoders": self.label_encoders,
            "caracteristicas": self.caracteristicas,
            "n_clusters": self.n_clusters
        }, ruta)
        print(f"\nModelo guardado en: {ruta}")
    
    def cargar_modelo(self, ruta="models/clustering_model.pkl"):
        datos = joblib.load(ruta)
        self.modelo = datos["modelo"]
        self.scaler = datos["scaler"]
        self.label_encoders = datos["label_encoders"]
        self.caracteristicas = datos["caracteristicas"]
        self.n_clusters = datos["n_clusters"]
        self.esta_entrenado = True
        print(f"Modelo cargado desde: {ruta}")


def main():
    from data.dataset_generator_unsupervised import GeneradorDatasetNoSupervisado
    
    print("\n" + "=" * 60)
    print("CLUSTERING - SISTEMA METRO MEDELLIN")
    print("=" * 60)
    
    generador = GeneradorDatasetNoSupervisado()
    
    print("\n1. CLUSTERING DE ESTACIONES")
    df_estaciones = generador.generar_dataset_estaciones(n_muestras=500)
    modelo_estaciones = ModeloClustering()
    resultado_est = modelo_estaciones.entrenar(df_estaciones, tipo="estaciones")
    modelo_estaciones.guardar_modelo("models/clustering_estaciones.pkl")
    
    print("\n2. CLUSTERING DE VIAJES")
    df_viajes = generador.generar_dataset_viajes_sin_etiquetas(n_muestras=800)
    modelo_viajes = ModeloClustering()
    resultado_viajes = modelo_viajes.entrenar(df_viajes, tipo="viajes")
    modelo_viajes.guardar_modelo("models/clustering_viajes.pkl")
    
    print("\n3. CLUSTERING DE PERFILES DE USUARIO")
    df_perfiles = generador.generar_dataset_perfiles_usuario(n_muestras=600)
    modelo_perfiles = ModeloClustering()
    resultado_perfiles = modelo_perfiles.entrenar(df_perfiles, tipo="perfiles")
    modelo_perfiles.guardar_modelo("models/clustering_perfiles.pkl")


if __name__ == "__main__":
    main()
