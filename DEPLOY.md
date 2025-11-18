# 🚀 Guía de Despliegue - Backend

## Desplegar en Render.com (Recomendado - Gratis)

### Paso 1: Preparar el Repositorio en GitHub

1. Ve a [GitHub](https://github.com) y crea un nuevo repositorio:
   - Nombre: `academic-risk-predictor-backend`
   - Visibilidad: Público o Privado
   - **NO** marques "Add README" ni ".gitignore" (ya los tienes)

2. En tu terminal, dentro de la carpeta `academic-risk-predictor-backend`:

```bash
git init
git add .
git commit -m "Initial commit: Backend API con ML"
git branch -M main
git remote add origin https://github.com/TU_USUARIO/academic-risk-predictor-backend.git
git push -u origin main
```

### Paso 2: Desplegar en Render

1. Ve a [Render.com](https://render.com) y crea una cuenta (gratis)

2. Click en "New +" → "Web Service"

3. Conecta tu repositorio de GitHub:
   - Autoriza Render para acceder a tu GitHub
   - Selecciona `academic-risk-predictor-backend`

4. Configuración del servicio:
   - **Name**: `academic-risk-predictor-api` (o el que prefieras)
   - **Environment**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `python main.py`
   - **Plan**: `Free`

5. Variables de entorno (opcional):
   - `PORT` → Se configura automáticamente por Render

6. Click en "Create Web Service"

7. Espera a que termine el despliegue (puede tomar 5-10 minutos la primera vez)

### Paso 3: Obtener la URL

Una vez desplegado, Render te dará una URL como:
```
https://academic-risk-predictor-api.onrender.com
```

**¡Guarda esta URL!** La necesitarás para configurar el frontend.

### Paso 4: Probar el API

Abre en tu navegador:
```
https://tu-api.onrender.com/health
```

Deberías ver:
```json
{
  "status": "healthy",
  "modelo_cargado": true,
  "scaler_cargado": true
}
```

### Paso 5: Documentación Automática

Accede a la documentación interactiva:
```
https://tu-api.onrender.com/docs
```

---

## Alternativa: Desplegar en Railway.app

### Paso 1: Subir a GitHub (mismo que arriba)

### Paso 2: Desplegar en Railway

1. Ve a [Railway.app](https://railway.app) y crea una cuenta

2. Click en "New Project" → "Deploy from GitHub repo"

3. Selecciona `academic-risk-predictor-backend`

4. Railway detectará automáticamente que es Python

5. El despliegue se inicia automáticamente

6. Una vez desplegado, ve a "Settings" → "Networking" → "Generate Domain"

7. Obtendrás una URL como:
```
https://academic-risk-predictor-backend-production.up.railway.app
```

---

## ⚠️ Notas Importantes

### Plan Gratuito de Render
- ✅ 750 horas gratis al mes
- ⚠️ El servicio se "duerme" después de 15 minutos de inactividad
- ⚠️ La primera petición después de dormir puede tardar ~30 segundos

### Plan Gratuito de Railway
- ✅ $5 USD de crédito gratis al mes
- ✅ No se duerme
- ⚠️ Límite de horas de uso

### Mantener el Servicio "Despierto" en Render

Si quieres evitar el tiempo de espera, puedes usar un servicio de ping:

1. [UptimeRobot](https://uptimerobot.com) (gratis)
2. Configurar un monitor HTTP cada 10 minutos a tu URL `/health`

---

## 🔧 Troubleshooting

### Error: "ModuleNotFoundError"
- Verifica que `requirements.txt` esté en el repositorio
- Build command debe ser: `pip install -r requirements.txt`

### Error: "Port already in use"
- Render asigna automáticamente el puerto
- Tu código ya está configurado para leer la variable `PORT`

### Error: "Application startup failed"
- Revisa los logs en el panel de Render
- Probablemente falta el dataset CSV

### El modelo no se carga
- Asegúrate de que `dataset_estudiantes_decimal.csv` esté en el repositorio
- El modelo se entrena automáticamente al iniciar

---

## 📊 Monitoreo

En el panel de Render/Railway puedes ver:
- Logs en tiempo real
- Uso de CPU y memoria
- Peticiones por minuto
- Errores

---

## 🔄 Actualizar el Código

Simplemente haz push a tu repositorio:

```bash
git add .
git commit -m "Actualización del backend"
git push
```

Render/Railway detectará el cambio y redesplegaráautomáticamente.

---

**¡Listo!** Tu backend está en producción 🎉

**Siguiente paso**: Desplegar el frontend y configurar la URL del backend.

