﻿# asr_server_sensesmall.py
from fastapi import FastAPI, File, UploadFile
import uvicorn
from funasr import AutoModel
import os
from pathlib import Path
import sys

# 将项目根目录添加到 sys.path，以便导入 common.config
PROJECT_ROOT = Path(__file__).resolve().parents[3]  # 向上4级到 26-home
sys.path.insert(0, str(PROJECT_ROOT))

from common.config import LANGUAGE  # 导入语言配置

# 设置模型缓存目录
CACHE_DIR = PROJECT_ROOT / "models" / "modelscope_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ["MODELSCOPE_CACHE"] = str(CACHE_DIR)

print(f"项目根目录: {PROJECT_ROOT}")
print(f"模型缓存目录: {CACHE_DIR}")
print(f"当前识别语言: {LANGUAGE}")

# 加载SenseVoiceSmall模型
print("正在加载SenseVoiceSmall模型...")
model = AutoModel(
    model="iic/SenseVoiceSmall",
    trust_remote_code=True,
    vad_model="fsmn-vad",  # VAD模型不区分语言
    device="cuda"          # 如果有GPU则使用，否则改为 "cpu"
)
print("模型加载完成！")

app = FastAPI()

@app.post("/api/speech_recognition")
async def speech_recognition(audio: UploadFile = File(...)):
    """
    语音识别接口，根据 config.LANGUAGE 自动切换中英文。
    """
    audio_data = await audio.read()
    # 根据全局配置设置识别语言
    lang = "en" if LANGUAGE == "en" else "zh"
    res = model.generate(
        input=audio_data,
        cache={},
        language=lang,      # 动态语言
        use_itn=True
    )
    text = res[0]["text"] if res else "识别失败"
    return {"code": 200, "text": text}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)