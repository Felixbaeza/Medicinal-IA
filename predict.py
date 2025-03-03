import joblib
import numpy as np

# Cargar modelo y escalador
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

def predecir_diabetes(datos_paciente):
    datos_paciente = np.array(datos_paciente).reshape(1, -1)
    datos_paciente = scaler.transform(datos_paciente)
    prediccion = model.predict(datos_paciente)
    return "Positivo para diabetes tipo 2" if prediccion[0] == 1 else "Negativo para diabetes tipo 2"

# Ejemplo de uso
if __name__ == "__main__":
    ejemplo_paciente = [2, 130, 70, 20, 79, 25.5, 0.5, 40]  # Datos de prueba
    resultado = predecir_diabetes(ejemplo_paciente)
    print("Predicción:", resultado)
