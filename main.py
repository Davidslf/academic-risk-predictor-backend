"""
Sistema de Predicción de Riesgo Académico
Backend FastAPI con Regresión Logística
Desarrollado como proyecto final - Semestre 2025-II
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import joblib
import os
from typing import Dict, List

# ============================================================================
# CONFIGURACIÓN DE LA APLICACIÓN
# ============================================================================

app = FastAPI(
    title="API de Predicción de Riesgo Académico",
    description="Sistema predictivo basado en Regresión Logística con análisis de IA",
    version="1.0.0"
)

# Configuración CORS - Permite acceso público desde cualquier origen
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Permite TODOS los orígenes (API pública)
    allow_credentials=False,  # Deshabilitado para permitir origins="*"
    allow_methods=["*"],  # Permite todos los métodos HTTP (GET, POST, etc.)
    allow_headers=["*"],  # Permite todos los headers
)

# Variables globales para el modelo y el scaler
model = None
scaler = None
promedio_estudiantes_aprobados = None

# ============================================================================
# MODELOS DE DATOS (PYDANTIC)
# ============================================================================

class EstudianteInput(BaseModel):
    """Modelo de entrada para los datos del estudiante"""
    promedio_asistencia: float = Field(..., ge=0, le=100, description="Porcentaje de asistencia (0-100)")
    promedio_seguimiento: float = Field(..., ge=0, le=5, description="Promedio de seguimiento (0-5)")
    nota_parcial_1: float = Field(..., ge=0, le=5, description="Nota del primer parcial (0-5)")
    inicios_sesion_plataforma: int = Field(..., ge=0, description="Número de inicios de sesión")
    uso_tutorias: int = Field(..., ge=0, le=10, description="Uso de tutorías (0-10)")

class ChatInput(BaseModel):
    """Modelo de entrada para el chat"""
    pregunta: str = Field(..., description="Pregunta del estudiante")
    datos_estudiante: EstudianteInput
    prediccion_actual: dict = Field(None, description="Predicción actual si existe")

    class Config:
        json_schema_extra = {
            "example": {
                "promedio_asistencia": 85.0,
                "promedio_seguimiento": 3.5,
                "nota_parcial_1": 3.2,
                "inicios_sesion_plataforma": 42,
                "uso_tutorias": 1
            }
        }

# ============================================================================
# FUNCIONES AUXILIARES
# ============================================================================

def cargar_o_entrenar_modelo():
    """
    Función que verifica si existen los archivos del modelo y scaler.
    Si no existen, los entrena desde cero con el dataset CSV.
    Si existen, los carga en memoria.
    """
    global model, scaler, promedio_estudiantes_aprobados
    
    modelo_path = "modelo_logistico.joblib"
    scaler_path = "scaler.joblib"
    dataset_path = "dataset_estudiantes_decimal.csv"
    
    # Si los archivos del modelo YA existen, solo cargarlos
    if os.path.exists(modelo_path) and os.path.exists(scaler_path):
        print("✅ Cargando modelo y scaler existentes...")
        model = joblib.load(modelo_path)
        scaler = joblib.load(scaler_path)
        print("✅ Modelo y scaler cargados exitosamente")
    else:
        # Si NO existen, entrenar desde cero
        print("🔄 Modelo no encontrado. Iniciando entrenamiento...")
        
        # Verificar que existe el dataset
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(
                f"❌ No se encontró el archivo de datos: {dataset_path}\n"
                f"Asegúrate de que el archivo CSV esté en el mismo directorio que main.py"
            )
        
        # 1. Cargar el dataset
        print(f"📊 Cargando dataset desde {dataset_path}...")
        df = pd.read_csv(dataset_path)
        print(f"✅ Dataset cargado: {len(df)} registros, {len(df.columns)} columnas")
        
        # 2. Preparar los datos (X, y)
        feature_columns = [
            'promedio_asistencia',
            'promedio_seguimiento',
            'nota_parcial_1',
            'inicios_sesion_plataforma',
            'uso_tutorias'
        ]
        
        X = df[feature_columns].values
        y = df['riesgo_reprobacion'].values
        
        print(f"📈 Datos preparados: X shape = {X.shape}, y shape = {y.shape}")
        
        # 3. Instanciar y ajustar el StandardScaler (Sección 5.3 del PDF)
        print("🔧 Escalando variables con StandardScaler...")
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        print("✅ Variables escaladas correctamente")
        
        # 4. Entrenar el modelo de Regresión Logística (Sección 4.1 del PDF)
        print("🤖 Entrenando modelo de Regresión Logística...")
        model = LogisticRegression(
            random_state=42,
            max_iter=1000,
            solver='lbfgs'
        )
        model.fit(X_scaled, y)
        print("✅ Modelo entrenado exitosamente")
        
        # 5. Guardar el modelo y el scaler para uso futuro
        print("💾 Guardando modelo y scaler...")
        joblib.dump(model, modelo_path)
        joblib.dump(scaler, scaler_path)
        print(f"✅ Modelo guardado en: {modelo_path}")
        print(f"✅ Scaler guardado en: {scaler_path}")
        
        # Mostrar información del modelo
        print("\n📊 Información del modelo entrenado:")
        print(f"   Intercepto (β₀): {model.intercept_[0]:.4f}")
        print(f"   Coeficientes (β₁...β₅):")
        for i, (col, coef) in enumerate(zip(feature_columns, model.coef_[0])):
            print(f"      {col}: {coef:.4f}")
    
    # Calcular promedios de estudiantes aprobados (riesgo = 0)
    print("\n📊 Calculando promedios de estudiantes aprobados...")
    df = pd.read_csv(dataset_path)
    estudiantes_aprobados = df[df['riesgo_reprobacion'] == 0]
    
    promedio_estudiantes_aprobados = {
        "promedio_asistencia": float(estudiantes_aprobados['promedio_asistencia'].mean()),
        "promedio_seguimiento": float(estudiantes_aprobados['promedio_seguimiento'].mean()),
        "nota_parcial_1": float(estudiantes_aprobados['nota_parcial_1'].mean()),
        "inicios_sesion_plataforma": float(estudiantes_aprobados['inicios_sesion_plataforma'].mean()),
        "uso_tutorias": float(estudiantes_aprobados['uso_tutorias'].mean())
    }
    
    print("✅ Promedios calculados:")
    for key, value in promedio_estudiantes_aprobados.items():
        print(f"   {key}: {value:.2f}")
    
    print("\n🚀 Sistema listo para realizar predicciones")


def generar_analisis_ia(datos_estudiante: Dict, probabilidad_riesgo: float) -> str:
    """
    Genera un análisis personalizado basado en patrones y reglas inteligentes.
    Sistema de consejería académica autónomo sin dependencias externas.
    
    Args:
        datos_estudiante: Diccionario con los datos del estudiante
        probabilidad_riesgo: Probabilidad de riesgo (0-1)
    
    Returns:
        Texto del análisis personalizado con consejos específicos
    """
    porcentaje_riesgo = probabilidad_riesgo * 100
    nivel_riesgo = "ALTO" if porcentaje_riesgo >= 70 else "MEDIO" if porcentaje_riesgo >= 40 else "BAJO"
    
    # Extraer datos del estudiante
    asistencia = datos_estudiante['promedio_asistencia']
    seguimiento = datos_estudiante['promedio_seguimiento']
    parcial = datos_estudiante['nota_parcial_1']
    logins = datos_estudiante['inicios_sesion_plataforma']
    usa_tutorias = datos_estudiante['uso_tutorias']
    
    # === ANÁLISIS INICIAL ===
    analisis = []
    
    # Determinar mensaje inicial según nivel de riesgo
    if nivel_riesgo == "ALTO":
        analisis.append("⚠️ **SITUACIÓN DE RIESGO ALTO**")
        analisis.append(f"Tu probabilidad de reprobación es del {porcentaje_riesgo:.1f}%. Esta situación requiere acciones inmediatas y concretas.")
    elif nivel_riesgo == "MEDIO":
        analisis.append("⚠️ **SITUACIÓN DE RIESGO MODERADO**")
        analisis.append(f"Tu probabilidad de reprobación es del {porcentaje_riesgo:.1f}%. Aún estás a tiempo de mejorar con cambios estratégicos.")
    else:
        analisis.append("✅ **SITUACIÓN FAVORABLE**")
        analisis.append(f"Tu probabilidad de reprobación es del {porcentaje_riesgo:.1f}%. Vas por buen camino, pero siempre hay espacio para mejorar.")
    
    analisis.append("")  # Línea en blanco
    
    # === IDENTIFICAR ÁREAS PROBLEMÁTICAS ===
    problemas = []
    fortalezas = []
    
    # Análisis de Asistencia
    if asistencia < 70:
        problemas.append(("asistencia", f"Tu asistencia ({asistencia:.1f}%) está muy por debajo del mínimo recomendado (85%)"))
    elif asistencia < 85:
        problemas.append(("asistencia", f"Tu asistencia ({asistencia:.1f}%) necesita mejorar para alcanzar el 85% recomendado"))
    else:
        fortalezas.append(f"Excelente asistencia ({asistencia:.1f}%)")
    
    # Análisis de Seguimiento
    if seguimiento < 2.5:
        problemas.append(("seguimiento", f"Tu seguimiento académico ({seguimiento:.1f}/5.0) es bajo, indica poca participación"))
    elif seguimiento < 3.5:
        problemas.append(("seguimiento", f"Tu seguimiento ({seguimiento:.1f}/5.0) puede mejorar con más participación activa"))
    else:
        fortalezas.append(f"Buen nivel de seguimiento ({seguimiento:.1f}/5.0)")
    
    # Análisis de Nota Parcial
    if parcial < 2.5:
        problemas.append(("parcial", f"Tu nota del parcial ({parcial:.1f}/5.0) está por debajo de 2.5, requiere refuerzo urgente"))
    elif parcial < 3.0:
        problemas.append(("parcial", f"Tu nota del parcial ({parcial:.1f}/5.0) necesita mejorar para aprobar con seguridad"))
    elif parcial < 3.5:
        problemas.append(("parcial", f"Tu nota del parcial ({parcial:.1f}/5.0) es aceptable pero mejorable"))
    else:
        fortalezas.append(f"Buen desempeño en el parcial ({parcial:.1f}/5.0)")
    
    # Análisis de Logins (Plataforma)
    if logins < 30:
        problemas.append(("logins", f"Tus inicios de sesión ({logins}) son muy bajos, indica poca interacción con el material"))
    elif logins < 40:
        problemas.append(("logins", f"Podrías aumentar tu uso de la plataforma (actualmente {logins} logins)"))
    else:
        fortalezas.append(f"Buen uso de la plataforma ({logins} logins)")
    
    # Análisis de Tutorías
    if usa_tutorias == 0:
        problemas.append(("tutorias", "No estás aprovechando el servicio de tutorías disponible"))
    else:
        fortalezas.append("Aprovechas las tutorías disponibles")
    
    # === MOSTRAR DIAGNÓSTICO ===
    if fortalezas:
        analisis.append("**🌟 Tus Fortalezas:**")
        for f in fortalezas:
            analisis.append(f"• {f}")
        analisis.append("")
    
    if problemas:
        analisis.append("**⚠️ Áreas que Requieren Atención:**")
        for tipo, desc in problemas:
            analisis.append(f"• {desc}")
        analisis.append("")
    
    # === GENERAR CONSEJOS PERSONALIZADOS ===
    analisis.append("**💡 Plan de Acción Personalizado:**")
    analisis.append("")
    
    consejos = []
    
    # Priorizar consejos según los problemas más graves
    problemas_dict = dict(problemas)
    
    # 1. Consejo sobre ASISTENCIA (máxima prioridad si hay problema)
    if "asistencia" in problemas_dict:
        if asistencia < 70:
            consejos.append({
                "prioridad": 1,
                "icono": "🎯",
                "titulo": "URGENTE: Mejora tu Asistencia",
                "descripcion": f"Actualmente tienes {asistencia:.1f}% de asistencia. Objetivo inmediato: llegar al 85%.",
                "acciones": [
                    "Organiza tu horario para no perderte ninguna clase",
                    "Si tienes problemas de transporte u otros, habla con tu coordinador",
                    f"Necesitas asistir consistentemente para recuperar terreno"
                ]
            })
        else:
            consejos.append({
                "prioridad": 2,
                "icono": "📅",
                "titulo": "Aumenta tu Asistencia",
                "descripcion": f"Con {asistencia:.1f}%, estás cerca del objetivo. ¡Un esfuerzo más!",
                "acciones": [
                    "Asiste a todas las clases las próximas semanas",
                    "Llega puntual para aprovechar toda la sesión"
                ]
            })
    
    # 2. Consejo sobre PARCIAL (crítico si está bajo)
    if "parcial" in problemas_dict:
        if parcial < 2.5:
            consejos.append({
                "prioridad": 1,
                "icono": "📚",
                "titulo": "URGENTE: Refuerza tus Conocimientos",
                "descripcion": f"Tu nota de {parcial:.1f}/5.0 indica dificultades con el contenido.",
                "acciones": [
                    "Solicita retroalimentación detallada del parcial",
                    "Identifica los temas específicos donde fallaste",
                    "Dedica al menos 2 horas diarias de estudio estructurado",
                    "Forma grupos de estudio con compañeros que dominen el tema"
                ]
            })
        elif parcial < 3.0:
            consejos.append({
                "prioridad": 2,
                "icono": "📝",
                "titulo": "Mejora tu Desempeño Académico",
                "descripcion": f"Tu {parcial:.1f}/5.0 es aprobatorio pero justo. Necesitas subir para el siguiente.",
                "acciones": [
                    "Revisa los errores del primer parcial",
                    "Practica con ejercicios similares a los del examen",
                    "Consulta con el profesor tus dudas específicas"
                ]
            })
        else:
            consejos.append({
                "prioridad": 3,
                "icono": "⭐",
                "titulo": "Mantén tu Rendimiento",
                "descripcion": f"Con {parcial:.1f}/5.0 vas bien. Mantén el nivel.",
                "acciones": [
                    "Continúa estudiando regularmente",
                    "Profundiza en temas más complejos"
                ]
            })
    
    # 3. Consejo sobre TUTORÍAS
    if "tutorias" in problemas_dict and porcentaje_riesgo >= 40:
        consejos.append({
            "prioridad": 2,
            "icono": "👨‍🏫",
            "titulo": "Aprovecha las Tutorías",
            "descripcion": "Las tutorías pueden marcar la diferencia en tu desempeño.",
            "acciones": [
                "Agenda sesiones de tutoría esta misma semana",
                "Prepara preguntas específicas antes de cada sesión",
                "Los estudiantes que usan tutorías tienen 45% más probabilidad de aprobar"
            ]
        })
    
    # 4. Consejo sobre SEGUIMIENTO/PARTICIPACIÓN
    if "seguimiento" in problemas_dict:
        if seguimiento < 2.5:
            consejos.append({
                "prioridad": 2,
                "icono": "🙋",
                "titulo": "Aumenta tu Participación",
                "descripcion": f"Tu seguimiento de {seguimiento:.1f}/5.0 indica poca interacción en clase.",
                "acciones": [
                    "Participa activamente haciendo preguntas",
                    "Completa todas las tareas y actividades a tiempo",
                    "Interactúa más con el profesor y compañeros"
                ]
            })
        else:
            consejos.append({
                "prioridad": 3,
                "icono": "💬",
                "titulo": "Mejora tu Participación",
                "descripcion": f"Con {seguimiento:.1f}/5.0 de seguimiento, puedes involucrarte más.",
                "acciones": [
                    "Participa al menos una vez por clase",
                    "Completa tareas extras si están disponibles"
                ]
            })
    
    # 5. Consejo sobre PLATAFORMA
    if "logins" in problemas_dict and porcentaje_riesgo >= 40:
        consejos.append({
            "prioridad": 3,
            "icono": "💻",
            "titulo": "Usa Más la Plataforma Educativa",
            "descripcion": f"Con {logins} logins, no estás aprovechando todos los recursos.",
            "acciones": [
                "Ingresa diariamente para revisar material nuevo",
                "Revisa videos, lecturas y recursos adicionales",
                "Haz los ejercicios de práctica disponibles"
            ]
        })
    
    # 6. Consejo GENERAL sobre organización
    if nivel_riesgo == "ALTO":
        consejos.append({
            "prioridad": 1,
            "icono": "🗓️",
            "titulo": "Crea un Plan de Recuperación",
            "descripcion": "Necesitas un cambio significativo en tu enfoque académico.",
            "acciones": [
                "Establece un horario fijo de estudio (mínimo 10 horas semanales)",
                "Elimina distracciones durante las horas de estudio",
                "Comunícate con tu profesor para acordar un plan de mejora",
                "Considera reducir horas de trabajo u otras actividades si es posible"
            ]
        })
    
    # Ordenar consejos por prioridad
    consejos.sort(key=lambda x: x["prioridad"])
    
    # Limitar a los 3-4 consejos más importantes
    consejos_mostrar = consejos[:4] if nivel_riesgo == "ALTO" else consejos[:3]
    
    # Formatear consejos
    for i, consejo in enumerate(consejos_mostrar, 1):
        analisis.append(f"**{consejo['icono']} {i}. {consejo['titulo']}**")
        analisis.append(consejo['descripcion'])
        for accion in consejo['acciones']:
            analisis.append(f"   • {accion}")
        analisis.append("")
    
    # === MENSAJE MOTIVACIONAL FINAL ===
    analisis.append("---")
    if nivel_riesgo == "ALTO":
        analisis.append("⚡ **Recuerda:** Aunque la situación es difícil, NO es imposible. Muchos estudiantes en tu situación han logrado recuperarse con esfuerzo consistente. ¡Tú también puedes!")
    elif nivel_riesgo == "MEDIO":
        analisis.append("💪 **Recuerda:** Estás a tiempo de cambiar el resultado. Con los ajustes correctos, puedes aprobar con buena nota.")
    else:
        analisis.append("🎉 **¡Excelente trabajo!** Mantén esta actitud y terminarás el curso exitosamente.")
    
    return "\n".join(analisis)


def calcular_detalles_matematicos(
    features_scaled: np.ndarray,
    probabilidad_riesgo: float
) -> Dict:
    """
    Calcula todos los detalles matemáticos de la predicción para transparencia.
    Implementa las fórmulas de la Sección 4.2 del PDF.
    
    Args:
        features_scaled: Array con las características escaladas
        probabilidad_riesgo: Probabilidad calculada por el modelo
    
    Returns:
        Diccionario con todos los detalles matemáticos
    """
    # Obtener coeficientes e intercepto del modelo
    coeficientes = model.coef_[0].tolist()
    intercepto = float(model.intercept_[0])
    
    # Calcular z (logit) = β₀ + Σ(βᵢ × xᵢ_scaled)
    # Esta es la fórmula lineal antes de aplicar la sigmoide
    z = intercepto + np.dot(features_scaled, model.coef_[0])
    valor_z = float(z)
    
    # Construir el cálculo paso a paso para mostrar al usuario
    terminos = [f"{intercepto:.4f}"]  # Intercepto
    
    feature_names = [
        "Asistencia",
        "Seguimiento",
        "Parcial 1",
        "Logins",
        "Tutorías"
    ]
    
    for i, (coef, feat_scaled, nombre) in enumerate(zip(coeficientes, features_scaled, feature_names)):
        impacto = coef * feat_scaled
        terminos.append(f"({coef:.4f} × {feat_scaled:.4f})")
    
    calculo_logit_texto = f"z = {' + '.join(terminos)} = {valor_z:.4f}"
    
    # Calcular probabilidad usando la función sigmoide: P = 1 / (1 + e^(-z))
    # Esta es la fórmula de la Sección 4.2 del PDF
    calculo_probabilidad_texto = f"P(riesgo) = 1 / (1 + e^(-{valor_z:.4f})) = {probabilidad_riesgo:.4f}"
    
    return {
        "formula_logit": r"z = \beta_0 + \sum_{i=1}^{n} (\beta_i \cdot x_i^{\text{scaled}})",
        "formula_sigmoide": r"P(\text{riesgo}) = \frac{1}{1 + e^{-z}}",
        "features_scaled": features_scaled.tolist(),
        "coeficientes": coeficientes,
        "intercepto": intercepto,
        "calculo_logit_texto": calculo_logit_texto,
        "valor_z": valor_z,
        "calculo_probabilidad_texto": calculo_probabilidad_texto
    }


# ============================================================================
# EVENTOS DEL CICLO DE VIDA
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """
    Evento que se ejecuta al iniciar la aplicación.
    Carga o entrena el modelo según sea necesario.
    """
    print("\n" + "="*80)
    print("🚀 INICIANDO SISTEMA DE PREDICCIÓN DE RIESGO ACADÉMICO")
    print("="*80 + "\n")
    
    cargar_o_entrenar_modelo()
    
    print("\n" + "="*80)
    print("✅ SISTEMA INICIADO Y LISTO PARA RECIBIR PETICIONES")
    print("="*80 + "\n")


# ============================================================================
# ENDPOINTS
# ============================================================================

@app.get("/")
async def root():
    """Endpoint raíz con información del API"""
    return {
        "mensaje": "API de Predicción de Riesgo Académico",
        "version": "1.0.0",
        "endpoints": {
            "POST /predict": "Realizar predicción de riesgo académico",
            "GET /health": "Verificar estado del servicio"
        },
        "documentacion": "/docs"
    }


@app.get("/health")
async def health_check():
    """Endpoint para verificar el estado del servicio"""
    return {
        "status": "healthy",
        "modelo_cargado": model is not None,
        "scaler_cargado": scaler is not None
    }


@app.post("/predict")
async def predecir_riesgo(estudiante: EstudianteInput):
    """
    Endpoint principal para predecir el riesgo de reprobación de un estudiante.
    
    Este endpoint realiza:
    1. Escalado de las características de entrada
    2. Predicción de probabilidad de riesgo
    3. Generación de análisis con IA (Gemini)
    4. Cálculo de detalles matemáticos completos
    5. Preparación de datos para gráfico de radar
    
    Args:
        estudiante: Datos del estudiante (asistencia, seguimiento, parcial, logins, tutorías)
    
    Returns:
        Diccionario con predicción, análisis IA, datos para radar y detalles matemáticos
    """
    try:
        # Verificar que el modelo y scaler estén cargados
        if model is None or scaler is None:
            raise HTTPException(
                status_code=500,
                detail="Modelo no inicializado. Reinicia el servidor."
            )
        
        # 1. Preparar los datos de entrada
        datos_estudiante = {
            "promedio_asistencia": estudiante.promedio_asistencia,
            "promedio_seguimiento": estudiante.promedio_seguimiento,
            "nota_parcial_1": estudiante.nota_parcial_1,
            "inicios_sesion_plataforma": estudiante.inicios_sesion_plataforma,
            "uso_tutorias": estudiante.uso_tutorias
        }
        
        # Convertir a array numpy
        X_input = np.array([[
            estudiante.promedio_asistencia,
            estudiante.promedio_seguimiento,
            estudiante.nota_parcial_1,
            estudiante.inicios_sesion_plataforma,
            estudiante.uso_tutorias
        ]])
        
        # 2. Escalar las características usando el scaler entrenado
        X_scaled = scaler.transform(X_input)
        
        # 3. Realizar la predicción de probabilidad
        # predict_proba devuelve [P(clase_0), P(clase_1)]
        # Queremos P(riesgo=1), que es la segunda columna
        probabilidad_riesgo = float(model.predict_proba(X_scaled)[0][1])
        
        # 4. Generar análisis personalizado basado en patrones
        print(f"\n🧠 Generando análisis personalizado para estudiante con {probabilidad_riesgo*100:.1f}% de riesgo...")
        analisis_ia = generar_analisis_ia(datos_estudiante, probabilidad_riesgo)
        
        # 5. Calcular detalles matemáticos completos
        detalles_matematicos = calcular_detalles_matematicos(
            X_scaled[0],
            probabilidad_riesgo
        )
        
        # 6. Preparar datos para el gráfico de radar
        datos_radar = {
            "labels": [
                "Asistencia (%)",
                "Seguimiento",
                "Parcial 1",
                "Logins",
                "Tutorías"
            ],
            "estudiante": [
                estudiante.promedio_asistencia,
                estudiante.promedio_seguimiento,
                estudiante.nota_parcial_1,
                estudiante.inicios_sesion_plataforma,
                estudiante.uso_tutorias
            ],
            "promedio_aprobado": [
                promedio_estudiantes_aprobados["promedio_asistencia"],
                promedio_estudiantes_aprobados["promedio_seguimiento"],
                promedio_estudiantes_aprobados["nota_parcial_1"],
                promedio_estudiantes_aprobados["inicios_sesion_plataforma"],
                promedio_estudiantes_aprobados["uso_tutorias"]
            ]
        }
        
        # 7. Construir y retornar la respuesta completa
        respuesta = {
            "probabilidad_riesgo": probabilidad_riesgo,
            "porcentaje_riesgo": probabilidad_riesgo * 100,
            "nivel_riesgo": (
                "ALTO" if probabilidad_riesgo >= 0.7 
                else "MEDIO" if probabilidad_riesgo >= 0.4 
                else "BAJO"
            ),
            "analisis_ia": analisis_ia,
            "datos_radar": datos_radar,
            "detalles_matematicos": detalles_matematicos
        }
        
        print(f"✅ Predicción completada: {probabilidad_riesgo*100:.2f}% de riesgo")
        
        return respuesta
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error en predicción: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error al procesar la predicción: {str(e)}"
        )


# ============================================================================
# PUNTO DE ENTRADA
# ============================================================================

@app.post("/chat")
async def chat_consejero(chat_input: ChatInput):
    """
    Endpoint para el chat inteligente con el consejero virtual.
    Responde preguntas personalizadas sobre el rendimiento académico del estudiante.
    """
    try:
        pregunta = chat_input.pregunta.lower()
        datos = chat_input.datos_estudiante
        prediccion = chat_input.prediccion_actual
        
        # Analizar la pregunta y generar respuesta
        respuesta = ""
        
        # Preguntas sobre cómo mejorar
        if any(palabra in pregunta for palabra in ["mejorar", "mejor", "subir", "aumentar"]):
            respuesta = f"""**💡 Cómo Mejorar Tu Rendimiento:**

Con tus datos actuales:
• Asistencia: {datos.promedio_asistencia}%
• Seguimiento: {datos.promedio_seguimiento}/5.0
• Nota Parcial 1: {datos.nota_parcial_1}/5.0
• Logins: {datos.inicios_sesion_plataforma}
• Tutorías: {datos.uso_tutorias}

**🎯 Áreas de Mejora:**

"""
            # Analizar cada área
            if datos.promedio_asistencia < 80:
                respuesta += "• **Asistencia:** Intenta llegar al 90% o más. Cada clase perdida es conocimiento que no recuperas fácilmente.\n"
            
            if datos.promedio_seguimiento < 3.5:
                respuesta += "• **Participación:** Participa más en clase, haz preguntas, completa tareas a tiempo.\n"
            
            if datos.nota_parcial_1 < 3.5:
                respuesta += "• **Notas:** Dedica más tiempo al estudio. Forma grupos de estudio, usa recursos adicionales.\n"
            
            if datos.inicios_sesion_plataforma < 20:
                respuesta += "• **Plataforma:** Ingresa más seguido (ideal: 3-4 veces por semana). Revisa materiales, foros, anuncios.\n"
            
            if datos.uso_tutorias < 3:
                respuesta += "• **Tutorías:** ¡Úsalas! Son gratuitas y te ayudan mucho. Intenta al menos 3-5 sesiones.\n"
                
            respuesta += "\n**⭐ Recomendación Principal:** Enfócate primero en tu área más débil y luego avanza a las demás."
        
        # Preguntas sobre áreas débiles
        elif any(palabra in pregunta for palabra in ["débil", "debil", "peor", "malo", "bajo", "baja"]):
            areas_debiles = []
            
            if datos.promedio_asistencia < 75:
                areas_debiles.append(("Asistencia", datos.promedio_asistencia, "%"))
            if datos.promedio_seguimiento < 3.0:
                areas_debiles.append(("Seguimiento", datos.promedio_seguimiento, "/5.0"))
            if datos.nota_parcial_1 < 3.0:
                areas_debiles.append(("Nota Parcial 1", datos.nota_parcial_1, "/5.0"))
            if datos.inicios_sesion_plataforma < 15:
                areas_debiles.append(("Uso de Plataforma", datos.inicios_sesion_plataforma, " logins"))
            if datos.uso_tutorias < 2:
                areas_debiles.append(("Uso de Tutorías", datos.uso_tutorias, " sesiones"))
            
            if areas_debiles:
                respuesta = "**⚠️ Tus Áreas Más Débiles:**\n\n"
                for i, (area, valor, unidad) in enumerate(areas_debiles, 1):
                    respuesta += f"{i}. **{area}:** {valor}{unidad}\n"
                respuesta += "\n💪 **Consejo:** Prioriza estas áreas en ese orden."
            else:
                respuesta = "**✅ ¡Excelente!** No tienes áreas particularmente débiles. Mantén tu buen desempeño en todas las áreas."
        
        # Preguntas sobre qué necesita para aprobar
        elif any(palabra in pregunta for palabra in ["aprobar", "pasar", "necesito", "requiero"]):
            if prediccion and prediccion.get("porcentaje_riesgo"):
                riesgo = prediccion["porcentaje_riesgo"]
                respuesta = f"""**📊 Análisis para Aprobar:**

Tu riesgo actual de reprobación es: **{riesgo:.1f}%**

"""
                if riesgo < 30:
                    respuesta += "✅ **Vas muy bien!** Con tu desempeño actual, tienes alta probabilidad de aprobar.\n\n**Recomendación:** Mantén tu nivel actual en todas las áreas."
                elif riesgo < 50:
                    respuesta += "⚠️ **Estás en la zona de riesgo medio.** Necesitas mejorar algunas áreas.\n\n**Para reducir tu riesgo:**\n"
                    respuesta += "• Sube tu asistencia a 85%+\n"
                    respuesta += "• Mejora tu nota del próximo parcial (objetivo: 3.5+)\n"
                    respuesta += "• Aumenta tu participación y seguimiento\n"
                else:
                    respuesta += "🚨 **Riesgo alto.** Necesitas acción inmediata.\n\n**Plan de Acción Urgente:**\n"
                    respuesta += "1. Habla con tu profesor HOY\n"
                    respuesta += "2. Asiste a TODAS las clases restantes\n"
                    respuesta += "3. Usa todas las tutorías disponibles\n"
                    respuesta += "4. Forma un grupo de estudio\n"
                    respuesta += "5. Dedica mínimo 2 horas diarias de estudio\n"
            else:
                respuesta = """**📝 Para Aprobar la Materia:**

Generalmente necesitas:
• **Asistencia:** Mínimo 80% (ideal: 90%+)
• **Notas:** Promedio de 3.0 o superior
• **Participación:** Activa y constante
• **Uso de recursos:** Plataforma y tutorías

💡 **Tip:** Haz una predicción primero para ver tu riesgo actual y obtener recomendaciones personalizadas."""
        
        # Preguntas sobre consejos generales
        elif any(palabra in pregunta for palabra in ["consejo", "recomend", "ayuda", "sugerencia"]):
            respuesta = f"""**🎓 Consejos Personalizados para Ti:**

**📊 Tu Situación Actual:**
• Asistencia: {datos.promedio_asistencia}% {'✅' if datos.promedio_asistencia >= 80 else '⚠️'}
• Seguimiento: {datos.promedio_seguimiento}/5.0 {'✅' if datos.promedio_seguimiento >= 3.5 else '⚠️'}
• Nota Parcial: {datos.nota_parcial_1}/5.0 {'✅' if datos.nota_parcial_1 >= 3.0 else '⚠️'}

**💪 Consejos Específicos:**

1. **Organízate:**
   • Crea un horario semanal de estudio
   • Dedica 1-2 horas diarias a esta materia
   
2. **Sé Constante:**
   • Ingresa a la plataforma 3-4 veces por semana
   • Revisa materiales antes y después de clase
   
3. **Busca Apoyo:**
   • Forma grupos de estudio con compañeros
   • Usa las tutorías (son gratis y efectivas)
   
4. **Participa Activamente:**
   • Haz preguntas en clase
   • Participa en foros y discusiones
   
5. **Prepárate Bien:**
   • Estudia con anticipación para los exámenes
   • Practica con ejercicios adicionales

**🎯 Objetivo:** Mejorar un poco cada semana. ¡Los pequeños cambios generan grandes resultados!"""
        
        # Pregunta general o no reconocida
        else:
            respuesta = f"""**🤖 Consejero Académico Virtual**

¡Hola! Estoy aquí para ayudarte a mejorar tu rendimiento académico.

**Tus datos actuales:**
• Asistencia: {datos.promedio_asistencia}%
• Seguimiento: {datos.promedio_seguimiento}/5.0
• Nota Parcial 1: {datos.nota_parcial_1}/5.0
• Logins: {datos.inicios_sesion_plataforma}
• Tutorías: {datos.uso_tutorias}

**Puedes preguntarme:**
• "¿Cómo puedo mejorar mi nota?"
• "¿Cuál es mi área más débil?"
• "¿Qué necesito para aprobar?"
• "Dame consejos personalizados"
• "¿Cómo usar las tutorías?"

¿En qué más puedo ayudarte? 😊"""
        
        return {"respuesta": respuesta}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error procesando la pregunta: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    
    # Obtener puerto desde variable de entorno (para despliegue) o usar 8000 por defecto
    port = int(os.getenv("PORT", 8000))
    
    print("\n" + "="*80)
    print("🎓 SISTEMA DE PREDICCIÓN DE RIESGO ACADÉMICO - BACKEND")
    print("="*80)
    print("📚 Proyecto Final - Semestre 2025-II")
    print("🤖 Tecnología: FastAPI + Regresión Logística + Chat Inteligente")
    print(f"🌐 Puerto: {port}")
    print("="*80 + "\n")
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=False,  # Desactivar reload en producción
        log_level="info"
    )

