import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import os

class ModeloPrediccionTiempo:
    def __init__(self):
        self.modelo = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.esta_entrenado = False
        
    def preprocesar_datos(self, df):
        df_procesado = df.copy()
        
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
        
        caracteristicas = [
            "distancia_km", "hora", "es_hora_pico", "es_finde_semana",
            "transferencias", "origen_enc", "destino_enc", 
            "tipo_transporte_enc", "dia_semana_enc", "clima_enc"
        ]
        
        X = df_procesado[caracteristicas]
        y = df_procesado["tiempo_viaje_min"]
        
        return X, y
    
    def entrenar(self, df, test_size=0.2, modelo_tipo="random_forest"):
        print("=" * 60)
        print("ENTRENAMIENTO: Modelo de Predicción de Tiempo de Viaje")
        print("=" * 60)
        
        X, y = self.preprocesar_datos(df)
        
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
        print(f"  MAE:  {mean_absolute_error(y_train, y_pred_train):.2f} min")
        print(f"  RMSE: {np.sqrt(mean_squared_error(y_train, y_pred_train)):.2f} min")
        print(f"  R²:   {r2_score(y_train, y_pred_train):.4f}")
        
        print("\nDatos de Prueba:")
        print(f"  MAE:  {mean_absolute_error(y_test, y_pred_test):.2f} min")
        print(f"  RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_test)):.2f} min")
        print(f"  R²:   {r2_score(y_test, y_pred_test):.4f}")
        
        self.esta_entrenado = True
        
        return y_test, y_pred_test
    
    def predecir(self, datos):
        if not self.esta_entrenado:
            raise ValueError("El modelo debe ser entrenado antes de predecir")
        
        df = pd.DataFrame([datos])
        
        df["origen_enc"] = self.label_encoders["origen"].transform([datos["origen"]])[0]
        df["destino_enc"] = self.label_encoders["destino"].transform([datos["destino"]])[0]
        df["tipo_transporte_enc"] = self.label_encoders["tipo_transporte"].transform([datos["tipo_transporte"]])[0]
        df["dia_semana_enc"] = self.label_encoders["dia_semana"].transform([datos["dia_semana"]])[0]
        df["clima_enc"] = self.label_encoders["clima"].transform([datos["clima"]])[0]
        
        caracteristicas = [
            "distancia_km", "hora", "es_hora_pico", "es_finde_semana",
            "transferencias", "origen_enc", "destino_enc",
            "tipo_transporte_enc", "dia_semana_enc", "clima_enc"
        ]
        
        X = df[caracteristicas]
        X_scaled = self.scaler.transform(X)
        
        return self.modelo.predict(X_scaled)[0]
    
    def guardar_modelo(self, ruta="models/tiempo_viaje_model.pkl"):
        os.makedirs(os.path.dirname(ruta), exist_ok=True)
        joblib.dump({
            "modelo": self.modelo,
            "scaler": self.scaler,
            "label_encoders": self.label_encoders
        }, ruta)
        print(f"\nModelo guardado en: {ruta}")
    
    def cargar_modelo(self, ruta="models/tiempo_viaje_model.pkl"):
        datos = joblib.load(ruta)
        self.modelo = datos["modelo"]
        self.scaler = datos["scaler"]
        self.label_encoders = datos["label_encoders"]
        self.esta_entrenado = True
        print(f"Modelo cargado desde: {ruta}")


def main():
    from data.dataset_generator import GeneradorDataset
    
    print("\n" + "=" * 60)
    print("PREDICCIÓN DE TIEMPO DE VIAJE - SISTEMA METRO MEDELLÍN")
    print("=" * 60 + "\n")
    
    generador = GeneradorDataset()
    df_viajes = generador.generar_dataset_viajes(n_muestras=5000)
    
    modelo = ModeloPrediccionTiempo()
    
    print("\n1. ENTRENAMIENTO CON RANDOM FOREST")
    y_test, y_pred = modelo.entrenar(df_viajes, modelo_tipo="random_forest")
    
    print("\n2. PREDICCIÓN DE EJEMPLO")
    ejemplo = {
        "origen": "San Antonio",
        "destino": "Poblado",
        "tipo_transporte": "Metro",
        "distancia_km": 5.0,
        "dia_semana": "Lunes",
        "hora": 8,
        "es_hora_pico": 1,
        "es_finde_semana": 0,
        "transferencias": 0,
        "clima": "Soleado"
    }
    
    tiempo_predicho = modelo.predecir(ejemplo)
    print(f"\nViaje: {ejemplo['origen']} -> {ejemplo['destino']}")
    print(f"Transporte: {ejemplo['tipo_transporte']}")
    print(f"Tiempo predicho: {tiempo_predicho:.1f} minutos")
    
    modelo.guardar_modelo()


if __name__ == "__main__":
    main()
