# ────────────────────────────────────────────────────────────────────────────────
# Imagen ligera basada en Python 3.11
FROM python:3.11-slim

# 1. Instalar dependencias del sistema necesarias para matplotlib / fontconfig
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        fontconfig \
        && rm -rf /var/lib/apt/lists/*

# 2. Crear usuario no-root para ejecutar la app
RUN useradd -m ctcxplorer
WORKDIR /home/ctcxplorer
# Matplotlib necesita un directorio de caché escribible
ENV MPLCONFIGDIR=/home/ctcxplorer/.cache/matplotlib
RUN mkdir -p $MPLCONFIGDIR && chown -R ctcxplorer:ctcxplorer $MPLCONFIGDIR

# 3. Copiar código y dependencias
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY ctcxplorer.py .

# 4. Puerto por defecto de Gradio
EXPOSE 7860

# 5. Comando de arranque
USER ctcxplorer
CMD ["python", "ctcxplorer.py"]
