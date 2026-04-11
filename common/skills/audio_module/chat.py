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

# ---------- 路径配置（动态获取，方便移植） ----------
BASE_DIR = Path(__file__).resolve().parent
AUDIO_DIR = BASE_DIR / "audio_cache"
CACHE_DIR = AUDIO_DIR / "cache"
MAX_FILES = 200

# 创建目录
for d in [AUDIO_DIR, CACHE_DIR]:
    d.mkdir(exist_ok=True)

VOICE = "en-US-JennyNeural"

# ---------- 辅助函数 ----------
def speed_to_rate(speed):
    percent = int((speed - 1.0) * 100)
    return f"{percent:+d}%"

def sanitize_filename(text):
    text = re.sub(r'[^\w\u4e00-\u9fff]', '', text)
    return text[:20]

def cleanup_old_files(directory: Path, max_files: int):
    """清理旧音频文件，保留最近 max_files 个"""
    files = list(directory.glob("*.mp3"))
    if len(files) <= max_files:
        return
    files.sort(key=lambda f: f.stat().st_mtime)
    for f in files[:-max_files]:
        f.unlink()
        print(f"清理旧文件: {f}")

# ---------- FastAPI 应用 ----------
app = FastAPI()

@app.post("/v1/audio/speech")
async def speech(
        model: str = Body("tts-1", embed=True),
        input: str = Body(..., embed=True),
        voice: str = Body("alloy", embed=True),
        speed: float = Body(1.0, embed=True)
):
    # 生成缓存 key
    key = hashlib.md5(f"{input}_{speed}".encode()).hexdigest()
    cache_path = CACHE_DIR / f"{key}.mp3"

    if cache_path.exists():
        print(f"使用缓存: {cache_path}")
        return Response(content=cache_path.read_bytes(), media_type="audio/mpeg")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = sanitize_filename(input)
    file_name = f"{timestamp}_{prefix}.mp3"
    audio_path = AUDIO_DIR / file_name

    rate_str = speed_to_rate(speed)
    try:
        communicate = edge_tts.Communicate(input, VOICE, rate=rate_str)
        await communicate.save(str(audio_path))  # edge_tts 需要字符串路径
    except Exception as e:
        print(f"音频生成失败: {e}")
        return {"error": str(e)}, 500

    # 复制到缓存目录
    shutil.copy2(str(audio_path), str(cache_path))

    # 清理旧文件
    cleanup_old_files(AUDIO_DIR, MAX_FILES)
    cleanup_old_files(CACHE_DIR, MAX_FILES // 2)

    return Response(content=audio_path.read_bytes(), media_type="audio/mpeg")

@app.get("/health")
async def health_check():
    return {"status": "success", "port": "8002", "module": "Edge-TTS with Cache"}

# ---------- 启动入口 ----------
if __name__ == "__main__":
    uvicorn.run(
        app=app,
        host="0.0.0.0",
        port=8002,
        log_level="info",
        access_log=False
    )