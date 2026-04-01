import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os

class ClasificadorTransporte:
    def __init__(self):
        self.modelo = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.clases = None
        self.esta_entrenado = False
        
    def preprocesar_datos(self, df):
        df_procesado = df.copy()
        
        le_origen = LabelEncoder()
        le_destino = LabelEncoder()
        le_dia = LabelEncoder()
        le_presupuesto = LabelEncoder()
        
        df_procesado["origen_enc"] = le_origen.fit_transform(df_procesado["origen"])
        df_procesado["destino_enc"] = le_destino.fit_transform(df_procesado["destino"])
        df_procesado["dia_semana_enc"] = le_dia.fit_transform(df_procesado["dia_semana"])
        df_procesado["presupuesto_enc"] = le_presupuesto.fit_transform(df_procesado["presupuesto"])
        
        self.label_encoders = {
            "origen": le_origen,
            "destino": le_destino,
            "dia_semana": le_dia,
            "presupuesto": le_presupuesto
        }
        
        caracteristicas = [
            "distancia_km", "hora", "es_hora_pico", "es_finde_semana",
            "es_urgente", "presupuesto_enc", "origen_enc", "destino_enc",
            "dia_semana_enc"
        ]
        
        X = df_procesado[caracteristicas]
        y = df_procesado["transporte_optimo"]
        
        return X, y, caracteristicas
    
    def entrenar(self, df, test_size=0.2, modelo_tipo="random_forest"):
        print("=" * 60)
        print("ENTRENAMIENTO: Clasificador de Transporte Óptimo")
        print("=" * 60)
        
        X, y, caracteristicas = self.preprocesar_datos(df)
        
        self.clases = y.unique()
        
        X_scaled = self.scaler.fit_transform(X)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=test_size, random_state=42, stratify=y
        )
        
        print(f"\nDatos de entrenamiento: {len(X_train)}")
        print(f"Datos de prueba: {len(X_test)}")
        print(f"Clases identificadas: {list(self.clases)}")
        
        if modelo_tipo == "random_forest":
            self.modelo = RandomForestClassifier(
                n_estimators=100,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )
        elif modelo_tipo == "gradient_boosting":
            self.modelo = GradientBoostingClassifier(
                n_estimators=100,
                max_depth=8,
                learning_rate=0.1,
                random_state=42
            )
        elif modelo_tipo == "decision_tree":
            self.modelo = DecisionTreeClassifier(
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42
            )
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
        print(f"  Accuracy: {accuracy_score(y_train, y_pred_train):.4f}")
        
        print("\nDatos de Prueba:")
        print(f"  Accuracy: {accuracy_score(y_test, y_pred_test):.4f}")
        
        print("\n" + "-" * 40)
        print("REPORTE DE CLASIFICACIÓN")
        print("-" * 40)
        print(classification_report(y_test, y_pred_test))
        
        print("-" * 40)
        print("MATRIZ DE CONFUSIÓN")
        print("-" * 40)
        cm = confusion_matrix(y_test, y_pred_test, labels=self.clases)
        print(f"Clases: {list(self.clases)}")
        print(cm)
        
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
        
        try:
            df["origen_enc"] = self.label_encoders["origen"].transform([datos["origen"]])[0]
        except ValueError:
            df["origen_enc"] = 0
        try:
            df["destino_enc"] = self.label_encoders["destino"].transform([datos["destino"]])[0]
        except ValueError:
            df["destino_enc"] = 0
        try:
            df["dia_semana_enc"] = self.label_encoders["dia_semana"].transform([datos["dia_semana"]])[0]
        except ValueError:
            df["dia_semana_enc"] = 0
        try:
            df["presupuesto_enc"] = self.label_encoders["presupuesto"].transform([datos["presupuesto"]])[0]
        except ValueError:
            df["presupuesto_enc"] = 1
        
        caracteristicas = [
            "distancia_km", "hora", "es_hora_pico", "es_finde_semana",
            "es_urgente", "presupuesto_enc", "origen_enc", "destino_enc",
            "dia_semana_enc"
        ]
        
        X = df[caracteristicas]
        X_scaled = self.scaler.transform(X)
        
        prediccion = self.modelo.predict(X_scaled)[0]
        probabilidades = self.modelo.predict_proba(X_scaled)[0]
        
        clases_ordenadas = self.modelo.classes_
        
        return {
            "transporte_predicho": prediccion,
            "probabilidades": dict(zip(clases_ordenadas, probabilidades))
        }
    
    def guardar_modelo(self, ruta="models/clasificador_transporte_model.pkl"):
        os.makedirs(os.path.dirname(ruta), exist_ok=True)
        joblib.dump({
            "modelo": self.modelo,
            "scaler": self.scaler,
            "label_encoders": self.label_encoders,
            "clases": self.clases
        }, ruta)
        print(f"\nModelo guardado en: {ruta}")
    
    def cargar_modelo(self, ruta="models/clasificador_transporte_model.pkl"):
        datos = joblib.load(ruta)
        self.modelo = datos["modelo"]
        self.scaler = datos["scaler"]
        self.label_encoders = datos["label_encoders"]
        self.clases = datos["clases"]
        self.esta_entrenado = True
        print(f"Modelo cargado desde: {ruta}")


def main():
    from data.dataset_generator import GeneradorDataset
    
    print("\n" + "=" * 60)
    print("CLASIFICADOR DE TRANSPORTE ÓPTIMO - SISTEMA METRO MEDELLÍN")
    print("=" * 60 + "\n")
    
    generador = GeneradorDataset()
    df_transporte = generador.generar_dataset_transporte(n_muestras=4000)
    
    modelo = ClasificadorTransporte()
    
    print("\n1. ENTRENAMIENTO CON RANDOM FOREST")
    y_test, y_pred = modelo.entrenar(df_transporte, modelo_tipo="random_forest")
    
    print("\n2. PREDICCIÓN DE EJEMPLO")
    ejemplo = {
        "origen": "San Antonio",
        "destino": "Poblado",
        "distancia_km": 5.0,
        "dia_semana": "Viernes",
        "hora": 18,
        "es_hora_pico": 1,
        "es_finde_semana": 0,
        "es_urgente": 0,
        "presupuesto": "medio"
    }
    
    resultado = modelo.predecir(ejemplo)
    print(f"\nViaje: {ejemplo['origen']} -> {ejemplo['destino']}")
    print(f"Distancia: {ejemplo['distancia_km']} km")
    print(f"Presupuesto: {ejemplo['presupuesto']}")
    print(f"Urgente: {'Sí' if ejemplo['es_urgente'] else 'No'}")
    print(f"\nTransporte recomendado: {resultado['transporte_predicho']}")
    print("\nProbabilidades:")
    for transporte, prob in sorted(resultado['probabilidades'].items(), key=lambda x: x[1], reverse=True):
        print(f"  {transporte}: {prob*100:.1f}%")
    
    modelo.guardar_modelo()


if __name__ == "__main__":
    main()
