FROM nvidia/cuda:12.3.2-cudnn9-runtime-ubuntu22.04

# Установим базовые зависимости
RUN apt-get update && apt-get install -y \
    python3 python3-pip ffmpeg libsndfile1 git && \
    rm -rf /var/lib/apt/lists/*

# Копируем файлы
WORKDIR /app
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

COPY handler.py .

# RunPod serverless запускает handler.py
CMD ["python3", "handler.py"]
