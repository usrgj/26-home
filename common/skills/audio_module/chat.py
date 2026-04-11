# -*- coding: utf-8 -*-
from fastapi import FastAPI, Body, Response
import uvicorn
import sys
import os
import asyncio
import edge_tts
from datetime import datetime
import re
import glob
import hashlib
import shutil
from pathlib import Path

# 解决中文乱码
sys.stdout.reconfigure(encoding='utf-8')

# 创建 FastAPI 实例（关键！）
app = FastAPI()

VOICE = "zh-CN-XiaoxiaoNeural"
BASE_DIR = os.environ.get("AUDIO_BASE_DIR", str(Path(__file__).resolve().parent))
AUDIO_DIR = os.path.join(BASE_DIR, "audio_cache")
CACHE_DIR = os.path.join(AUDIO_DIR, "cache")
MAX_FILES = 200

# 创建目录
for d in [AUDIO_DIR, CACHE_DIR]:
    os.makedirs(d, exist_ok=True)

def speed_to_rate(speed):
    percent = int((speed - 1.0) * 100)
    return f"{percent:+d}%"

def sanitize_filename(text):
    text = re.sub(r'[^\w\u4e00-\u9fff]', '', text)
    return text[:20]

def cleanup_old_files(directory, max_files):
    files = glob.glob(os.path.join(directory, "*.mp3"))
    if len(files) <= max_files:
        return
    files.sort(key=lambda f: os.path.getmtime(f))
    for f in files[:-max_files]:
        os.remove(f)
        print(f"清理旧文件: {f}")

@app.post("/v1/audio/speech")
async def speech(
        model: str = Body("tts-1", embed=True),
        input: str = Body(..., embed=True),
        voice: str = Body("alloy", embed=True),
        speed: float = Body(1.0, embed=True)
):
    # 生成缓存 key
    key = hashlib.md5(f"{input}_{speed}".encode()).hexdigest()
    cache_path = os.path.join(CACHE_DIR, f"{key}.mp3")

    if os.path.exists(cache_path):
        print(f"使用缓存: {cache_path}")
        return Response(content=open(cache_path, "rb").read(), media_type="audio/mpeg")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = sanitize_filename(input)
    file_name = f"{timestamp}_{prefix}.mp3"
    audio_path = os.path.join(AUDIO_DIR, file_name)

    rate_str = speed_to_rate(speed)
    try:
        communicate = edge_tts.Communicate(input, VOICE, rate=rate_str)
        await communicate.save(audio_path)
    except Exception as e:
        print(f"音频生成失败: {e}")
        return {"error": str(e)}, 500

    shutil.copy2(audio_path, cache_path)

    cleanup_old_files(AUDIO_DIR, MAX_FILES)
    cleanup_old_files(CACHE_DIR, MAX_FILES // 2)

    return Response(content=open(audio_path, "rb").read(), media_type="audio/mpeg")

@app.get("/health")
async def health_check():
    return {"status": "success", "port": "8002", "module": "Edge-TTS with Cache"}

if __name__ == "__main__":
    uvicorn.run(
        app=app,
        host="0.0.0.0",
        port=8002,
        log_level="info",
        access_log=False
    )
