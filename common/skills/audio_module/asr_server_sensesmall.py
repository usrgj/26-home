# asr_server_sensesmall.py
from fastapi import FastAPI, File, UploadFile
import uvicorn
from funasr import AutoModel
import os
from pathlib import Path

# 获取项目根目录（向上4级）
PROJECT_ROOT = Path(__file__).resolve().parents[3]  # 因为当前文件在 .../common/skills/audio_module/
CACHE_DIR = PROJECT_ROOT / "models" / "modelscope_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ["MODELSCOPE_CACHE"] = str(CACHE_DIR)
print(PROJECT_ROOT)

# 加载SenseVoiceSmall模型
print("正在加载SenseVoiceSmall模型...")
model = AutoModel(
    model="iic/SenseVoiceSmall",
    trust_remote_code=True,
    vad_model="fsmn-vad",  # 可选，启用VAD提高准确率
    device="cuda"
)
print("模型加载完成！")

app = FastAPI()

@app.post("/api/speech_recognition")
async def speech_recognition(audio: UploadFile = File(...)):
    """
    语音识别接口

    接收上传的音频文件，调用模型进行语音识别并返回结果。

    Args:
        audio (UploadFile): 上传的音频文件对象

    Returns:
        dict: 包含状态码和识别文本的响应字典
            - code (int): HTTP 状态码
            - text (str): 识别出的文本内容
    """
    audio_data = await audio.read()
    # 调用模型进行语音识别生成
    res = model.generate(
        input=audio_data,
        cache={},
        language="en",  # 自动识别语言
        use_itn=True      # 使用逆文本标准化
    )
    # 解析识别结果并处理异常情况
    text = res[0]["text"] if res else "识别失败"
    return {"code": 200, "text": text}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)