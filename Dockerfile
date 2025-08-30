FROM nvidia/cuda:12.3.2-cudnn9-runtime-ubuntu22.04

# Установим базовые зависимости
RUN apt-get update && apt-get install -y \
    python3 python3-pip ffmpeg libsndfile1 git && \
    rm -rf /var/lib/apt/lists/*

# Рабочая директория
WORKDIR /app

# Копируем зависимости
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Копируем код
COPY server.py .

# Запускаем FastAPI через uvicorn
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
