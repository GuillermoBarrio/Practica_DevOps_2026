# Usa una imagen oficial de Python ligera
FROM python:3.11-slim

# Evita que Python escriba archivos .pyc y asegura que los logs salgan directos
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Establece el directorio de trabajo dentro del contenedor
WORKDIR /app

# Instala las dependencias del sistema necesarias para compilar librerías científicas
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    gcc \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Copia primero el requirements.txt para aprovechar la cache de Docker
COPY requirements.txt .

# Actualiza pip e instala las librerías de Python
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copia el resto del código de la aplicación (incluyendo el CSV)
COPY . .

# Expone el puerto por defecto que usa Google Cloud Run ($PORT)
EXPOSE 8080

# Comando para arrancar Streamlit configurado para Cloud Run
CMD ["sh", "-c", "streamlit run app-comentario.py --server.port=${PORT:-8080} --server.address=0.0.0.0 --server.headless=true --server.enableCORS=false --server.enableXsrfProtection=false"]