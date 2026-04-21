# chat.py
import edge_tts
import uvicorn
from fastapi import FastAPI, Body, Response
import os
import hashlib
import shutil
import glob
from datetime import datetime
import sys
from pathlib import Path

# 将项目根目录添加到 sys.path，以便导入 common.config
PROJECT_ROOT = Path(__file__).resolve().parents[3]  # 向上4级到 26-home
sys.path.insert(0, str(PROJECT_ROOT))

from common.config import LANGUAGE

app = FastAPI()

# 根据全局配置选择默认语音
if LANGUAGE == "en":
    VOICE_DEFAULT = "en-US-JennyNeural"   # 英文女声
else:
    VOICE_DEFAULT = "zh-CN-XiaoxiaoNeural" # 中文女声

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
AUDIO_DIR = os.path.join(BASE_DIR, "audio_cache")
CACHE_DIR = AUDIO_DIR   # 统一使用 audio_cache 目录
os.makedirs(AUDIO_DIR, exist_ok=True)

def speed_to_rate(speed):
    percent = int((speed - 1.0) * 100)
    return f"{percent:+d}%"

@app.post("/v1/audio/speech")
async def speech(
    model: str = Body("tts-1", embed=True),
    input: str = Body(..., embed=True),
    voice: str = Body(VOICE_DEFAULT, embed=True),
    speed: float = Body(1.0, embed=True)
):
    # 生成缓存 key（包含语言信息）
    key = hashlib.md5(f"{input}_{speed}_{voice}".encode()).hexdigest()
    cache_path = os.path.join(CACHE_DIR, f"{key}.mp3")
    if os.path.exists(cache_path):
        return Response(content=open(cache_path, "rb").read(), media_type="audio/mpeg")

    # 使用传入的 voice 参数（若未传则用默认语音）
    communicate = edge_tts.Communicate(input, voice, rate=speed_to_rate(speed))
    audio_path = os.path.join(AUDIO_DIR, f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp3")
    try:
        await communicate.save(audio_path)
    except Exception as e:
        return Response(content=f"TTS failed: {e}", status_code=500)
    
    # 复制到缓存（现在 cache_path 也在同一个 audio_cache 目录）
    shutil.copy2(audio_path, cache_path)
    # 可选：清理旧文件（保留最近200个）
    files = glob.glob(os.path.join(AUDIO_DIR, "*.mp3"))
    if len(files) > 200:
        files.sort(key=os.path.getmtime)
        for f in files[:-200]:
            os.remove(f)
    # 注意：CACHE_DIR 和 AUDIO_DIR 相同，不需要重复清理，否则会重复删除
    # 但为了避免重复，可以只清理一次；上面的清理已经包含所有 mp3 文件
    
    return Response(content=open(audio_path, "rb").read(), media_type="audio/mpeg")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8002)
