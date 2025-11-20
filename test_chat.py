import requests
import json

def test_chat():
    url = "http://localhost:8000/chat"
    
    payload = {
        "pregunta": "¿Qué necesito para aprobar la materia?",
        "datos_estudiante": {
            "promedio_asistencia": 85.0,
            "promedio_seguimiento": 3.5,
            "nota_parcial_1": 3.2,
            "inicios_sesion_plataforma": 42,
            "uso_tutorias": 1
        },
        "prediccion_actual": {
            "porcentaje_riesgo": 45.0
        }
    }
    
    print("Enviando pregunta al chatbot...")
    print(f"Pregunta: {payload['pregunta']}")
    
    try:
        response = requests.post(url, json=payload)
        
        if response.status_code == 200:
            data = response.json()
            print("\nRespuesta del Chatbot (Gemini):")
            print("-" * 50)
            print(data["respuesta"])
            print("-" * 50)
        else:
            print(f"Error: {response.status_code}")
            print(response.text)
            
    except requests.exceptions.ConnectionError:
        print("Error: No se pudo conectar al servidor. Asegúrate de que 'main.py' esté ejecutándose.")

if __name__ == "__main__":
    test_chat()
