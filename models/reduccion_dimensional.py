import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import joblib
import os

class ModeloReduccionDimensional:
    def __init__(self):
        self.pca = None
        self.tsne = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.caracteristicas_originales = None
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
            
            self.caracteristicas_originales = [
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
            
            self.caracteristicas_originales = [
                "distancia_km", "tiempo_viaje_min", "costo",
                "transferencias", "es_hora_pico", "es_finde_semana",
                "ocupacion_estimada"
            ]
        
        X = df_procesado[self.caracteristicas_originales]
        
        return X
    
    def entrenar_pca(self, X, n_componentes=None):
        print("\n" + "=" * 50)
        print("PCA - ANALISIS DE COMPONENTES PRINCIPALES")
        print("=" * 50)
        
        if n_componentes is None:
            n_componentes = min(2, X.shape[1])
        
        X_scaled = self.scaler.fit_transform(X)
        
        self.pca = PCA(n_components=n_componentes)
        X_pca = self.pca.fit_transform(X_scaled)
        
        varianza_explicada = self.pca.explained_variance_ratio_
        varianza_acumulada = np.cumsum(varianza_explicada)
        
        print(f"\nNumero de componentes: {n_componentes}")
        print(f"Caracteristicas originales: {len(self.caracteristicas_originales)}")
        
        print("\n" + "-" * 50)
        print("VARIANZA EXPLICADA POR COMPONENTE")
        print("-" * 50)
        for i, (var, var_acum) in enumerate(zip(varianza_explicada, varianza_acumulada)):
            print(f"  PC{i+1}: {var*100:.2f}% (Acumulada: {var_acum*100:.2f}%)")
        
        print("\n" + "-" * 50)
        print("CARGA DE CARACTERISTICAS EN COMPONENTES")
        print("-" * 50)
        
        df_cargas = pd.DataFrame(
            self.pca.components_.T,
            columns=[f'PC{i+1}' for i in range(n_componentes)],
            index=self.caracteristicas_originales
        )
        
        print("\nCargas PC1:")
        pc1_sorted = df_cargas['PC1'].abs().sort_values(ascending=False)
        for feat in pc1_sorted.head(5).index:
            print(f"  {feat}: {df_cargas.loc[feat, 'PC1']:.4f}")
        
        print("\nCargas PC2:")
        if 'PC2' in df_cargas.columns:
            pc2_sorted = df_cargas['PC2'].abs().sort_values(ascending=False)
            for feat in pc2_sorted.head(5).index:
                print(f"  {feat}: {df_cargas.loc[feat, 'PC2']:.4f}")
        
        n_comp_optimo = np.argmax(varianza_acumulada >= 0.95) + 1
        print(f"\nComponentes para 95% de varianza: {n_comp_optimo}")
        
        return X_pca
    
    def entrenar_tsne(self, X, n_componentes=2, perplexity=30):
        print("\n" + "=" * 50)
        print("t-SNE - REDUCCION NO LINEAL")
        print("=" * 50)
        
        X_scaled = self.scaler.fit_transform(X)
        
        self.tsne = TSNE(
            n_components=n_componentes,
            perplexity=perplexity,
            random_state=42,
            max_iter=1000
        )
        
        X_tsne = self.tsne.fit_transform(X_scaled)
        
        print(f"\nDatos reducidos a {n_componentes} dimensiones")
        print(f"Perplexity: {perplexity}")
        
        return X_tsne
    
    def transformar(self, X, metodo="pca"):
        X_scaled = self.scaler.transform(X)
        
        if metodo == "pca":
            return self.pca.transform(X_scaled)
        elif metodo == "tsne":
            return self.tsne.fit(X_scaled)
        else:
            raise ValueError(f"Metodo no reconocido: {metodo}")
    
    def analisis_completo(self, df, tipo="estaciones"):
        print("=" * 60)
        print(f"ANALISIS DE REDUCCION DIMENSIONAL - {tipo.upper()}")
        print("=" * 60)
        
        X = self.preprocesar_datos(df, tipo)
        
        print(f"\nDatos originales: {X.shape[0]} muestras, {X.shape[1]} caracteristicas")
        
        X_pca = self.entrenar_pca(X, n_componentes=3)
        
        X_tsne = self.entrenar_tsne(X, n_componentes=2, perplexity=30)
        
        print("\n" + "=" * 50)
        print("COMPARACION METODOS")
        print("=" * 50)
        print(f"  PCA: Proyeccion lineal, preserva varianza global")
        print(f"  t-SNE: Proyeccion no lineal, preserva estructura local")
        print(f"  Uso recomendado: Visualizacion y descubrimiento de patrones")
        
        self.esta_entrenado = True
        
        return {
            "X_pca": X_pca,
            "X_tsne": X_tsne,
            "varianza_explicada": self.pca.explained_variance_ratio_
        }
    
    def guardar_modelo(self, ruta="models/reduccion_dimensional_model.pkl"):
        os.makedirs(os.path.dirname(ruta), exist_ok=True)
        joblib.dump({
            "pca": self.pca,
            "tsne": self.tsne,
            "scaler": self.scaler,
            "label_encoders": self.label_encoders,
            "caracteristicas_originales": self.caracteristicas_originales
        }, ruta)
        print(f"\nModelo guardado en: {ruta}")
    
    def cargar_modelo(self, ruta="models/reduccion_dimensional_model.pkl"):
        datos = joblib.load(ruta)
        self.pca = datos["pca"]
        self.tsne = datos["tsne"]
        self.scaler = datos["scaler"]
        self.label_encoders = datos["label_encoders"]
        self.caracteristicas_originales = datos["caracteristicas_originales"]
        self.esta_entrenado = True
        print(f"Modelo cargado desde: {rura}")


def main():
    from data.dataset_generator_unsupervised import GeneradorDatasetNoSupervisado
    
    print("\n" + "=" * 60)
    print("REDUCCION DIMENSIONAL - SISTEMA METRO MEDELLIN")
    print("=" * 60)
    
    generador = GeneradorDatasetNoSupervisado()
    
    print("\n1. ANALISIS DE ESTACIONES")
    df_estaciones = generador.generar_dataset_estaciones(n_muestras=500)
    modelo_est = ModeloReduccionDimensional()
    modelo_est.analisis_completo(df_estaciones, tipo="estaciones")
    modelo_est.guardar_modelo("models/reduccion_dimensional_estaciones.pkl")
    
    print("\n2. ANALISIS DE VIAJES")
    df_viajes = generador.generar_dataset_viajes_sin_etiquetas(n_muestras=400)
    modelo_viajes = ModeloReduccionDimensional()
    modelo_viajes.analisis_completo(df_viajes, tipo="viajes")
    modelo_viajes.guardar_modelo("models/reduccion_dimensional_viajes.pkl")


if __name__ == "__main__":
    main()
