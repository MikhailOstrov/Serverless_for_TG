import runpod
import base64
import io
import requests
import soundfile as sf
from faster_whisper import WhisperModel
from huggingface_hub import snapshot_download
import os
from dotenv import load_dotenv

load_dotenv() 

ASR_MODEL_NAME = "deepdml/faster-whisper-large-v3-turbo-ct2"
HF_CACHE = "/workspace/.cache/huggingface"
HF_TOKEN = os.getenv("HF_TOKEN")

def load_model():
    print(f"Проверка и загрузка модели {ASR_MODEL_NAME}...")
    local_path = snapshot_download(
        repo_id=ASR_MODEL_NAME,
        cache_dir=HF_CACHE,
        local_files_only=False,
        token=HF_TOKEN
    )
    print(f"Модель скачана в {local_path}, загружаем...")
    return WhisperModel(local_path, device="cuda", compute_type="float16")

print(">>> Загружаем модель...")
model = load_model()
print(">>> Модель готова.")


def transcribe(audio_bytes: bytes) -> str:
    """Функция транскрибации"""
    audio_stream = io.BytesIO(audio_bytes)
    audio, sr = sf.read(audio_stream, dtype="float32")
    segments, _ = model.transcribe(audio, beam_size=3, best_of=1, vad_filter=False, language="ru")
    return " ".join(seg.text for seg in segments).strip()


def handler(job):
    try:
        data = job["input"]

        # Получение аудио
        if "audio_url" in data:
            resp = requests.get(data["audio_url"])
            audio_bytes = resp.content
        else:
            audio_bytes = base64.b64decode(data["audio_b64"])

        text = transcribe(audio_bytes)

        return {"status": "ok", "chat_id": data.get("chat_id"), "text": text}
    except Exception as e:
        return {"status": "error", "error": str(e)}


runpod.serverless.start({"handler": handler})
