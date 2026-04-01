import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import os

class ModeloPrediccionDemanda:
    def __init__(self):
        self.modelo = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.esta_entrenado = False
        
    def preprocesar_datos(self, df):
        df_procesado = df.copy()
        
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
        
        caracteristicas = [
            "estacion_enc", "hora", "es_hora_pico", "es_finde_semana",
            "dia_semana_enc", "clima_enc", "evento_especial"
        ]
        
        caracteristicas = [
            "estacion_enc", "hora", "es_hora_pico", "es_finde_semana",
            "dia_semana_enc", "clima_enc", "evento_especial"
        ]
        
        X = df_procesado[caracteristicas]
        y = df_procesado["afluencia_promedio"]
        
        return X, y, caracteristicas
    
    def entrenar(self, df, test_size=0.2, modelo_tipo="random_forest"):
        print("=" * 60)
        print("ENTRENAMIENTO: Modelo de Predicción de Demanda/Afluencia")
        print("=" * 60)
        
        X, y, caracteristicas = self.preprocesar_datos(df)
        
        X_scaled = self.scaler.fit_transform(X)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=test_size, random_state=42
        )
        
        print(f"\nDatos de entrenamiento: {len(X_train)}")
        print(f"Datos de prueba: {len(X_test)}")
        
        if modelo_tipo == "random_forest":
            self.modelo = RandomForestRegressor(
                n_estimators=100,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )
        elif modelo_tipo == "gradient_boosting":
            self.modelo = GradientBoostingRegressor(
                n_estimators=100,
                max_depth=8,
                learning_rate=0.1,
                random_state=42
            )
        elif modelo_tipo == "linear":
            self.modelo = LinearRegression()
        else:
            raise ValueError(f"Tipo de modelo no reconocido: {modelo_tipo}")
        
        print(f"\nModelo: {modelo_tipo}")
        print("Entrenando...")
        
        self.modelo.fit(X_train, y_train)
        
        y_pred_train = self.modelo.predict(X_train)
        y_pred_test = self.modelo.predict(X_test)
        
        print("\n" + "-" * 40)
        print("RESULTADOS ENTRENAMIENTO")
        print("-" * 40)
        print("\nDatos de Entrenamiento:")
        print(f"  MAE:  {mean_absolute_error(y_train, y_pred_train):.2f} pasajeros")
        print(f"  RMSE: {np.sqrt(mean_squared_error(y_train, y_pred_train)):.2f}")
        print(f"  R²:   {r2_score(y_train, y_pred_train):.4f}")
        
        print("\nDatos de Prueba:")
        print(f"  MAE:  {mean_absolute_error(y_test, y_pred_test):.2f} pasajeros")
        print(f"  RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_test)):.2f}")
        print(f"  R²:   {r2_score(y_test, y_pred_test):.4f}")
        
        importancia = pd.DataFrame({
            "caracteristica": caracteristicas,
            "importancia": self.modelo.feature_importances_
        }).sort_values("importancia", ascending=False)
        
        print("\n" + "-" * 40)
        print("IMPORTANCIA DE CARACTERÍSTICAS")
        print("-" * 40)
        for _, row in importancia.iterrows():
            print(f"  {row['caracteristica']}: {row['importancia']:.4f}")
        
        self.esta_entrenado = True
        
        return y_test, y_pred_test
    
    def predecir(self, datos):
        if not self.esta_entrenado:
            raise ValueError("El modelo debe ser entrenado antes de predecir")
        
        df = pd.DataFrame([datos])
        
        df["estacion_enc"] = self.label_encoders["estacion"].transform([datos["estacion"]])[0]
        df["dia_semana_enc"] = self.label_encoders["dia_semana"].transform([datos["dia_semana"]])[0]
        df["clima_enc"] = self.label_encoders["clima"].transform([datos["clima"]])[0]
        
        caracteristicas = [
            "estacion_enc", "hora", "es_hora_pico", "es_finde_semana",
            "dia_semana_enc", "clima_enc", "evento_especial"
        ]
        
        X = df[caracteristicas]
        X_scaled = self.scaler.transform(X)
        
        return self.modelo.predict(X_scaled)[0]
    
    def predecir_ocupacion(self, datos):
        afluenica = self.predecir(datos)
        capacidad = 5000
        ocupacion = (afluenica / capacidad) * 100
        
        if ocupacion > 80:
            nivel = "Alto"
        elif ocupacion > 50:
            nivel = "Medio"
        else:
            nivel = "Bajo"
        
        return {
            "afluencia_predicha": round(afluenica),
            "porcentaje_ocupacion": round(ocupacion, 1),
            "nivel_congestion": nivel
        }
    
    def guardar_modelo(self, ruta="models/demanda_model.pkl"):
        os.makedirs(os.path.dirname(ruta), exist_ok=True)
        joblib.dump({
            "modelo": self.modelo,
            "scaler": self.scaler,
            "label_encoders": self.label_encoders
        }, ruta)
        print(f"\nModelo guardado en: {ruta}")
    
    def cargar_modelo(self, ruta="models/demanda_model.pkl"):
        datos = joblib.load(ruta)
        self.modelo = datos["modelo"]
        self.scaler = datos["scaler"]
        self.label_encoders = datos["label_encoders"]
        self.esta_entrenado = True
        print(f"Modelo cargado desde: {ruta}")


def main():
    from data.dataset_generator import GeneradorDataset
    
    print("\n" + "=" * 60)
    print("PREDICCIÓN DE DEMANDA - SISTEMA METRO MEDELLÍN")
    print("=" * 60 + "\n")
    
    generador = GeneradorDataset()
    df_afluencia = generador.generar_dataset_afluencia(n_muestras=3000)
    
    modelo = ModeloPrediccionDemanda()
    
    print("\n1. ENTRENAMIENTO CON RANDOM FOREST")
    y_test, y_pred = modelo.entrenar(df_afluencia, modelo_tipo="random_forest")
    
    print("\n2. PREDICCIÓN DE EJEMPLO")
    ejemplo = {
        "estacion": "San Antonio",
        "dia_semana": "Martes",
        "hora": 8,
        "es_hora_pico": 1,
        "es_finde_semana": 0,
        "clima": "Soleado",
        "evento_especial": 0
    }
    
    resultado = modelo.predecir_ocupacion(ejemplo)
    print(f"\nEstación: {ejemplo['estacion']}")
    print(f"Fecha: {ejemplo['dia_semana']} a las {ejemplo['hora']}:00")
    print(f"Afluencia predicha: {resultado['afluencia_predicha']} pasajeros")
    print(f"Ocupación: {resultado['porcentaje_ocupacion']}%")
    print(f"Nivel de congestión: {resultado['nivel_congestion']}")
    
    modelo.guardar_modelo()


if __name__ == "__main__":
    main()
