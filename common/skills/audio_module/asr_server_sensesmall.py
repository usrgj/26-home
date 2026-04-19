# asr_server_sensesmall.py
from fastapi import FastAPI, File, UploadFile
import uvicorn
from funasr import AutoModel
import os
from pathlib import Path
import sys

# 将项目根目录添加到 sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from common.config import LANGUAGE, COMMON_DRINKS

# 设置模型缓存目录
CACHE_DIR = PROJECT_ROOT / "models" / "modelscope_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ["MODELSCOPE_CACHE"] = str(CACHE_DIR)

print(f"项目根目录: {PROJECT_ROOT}")
print(f"模型缓存目录: {CACHE_DIR}")
print(f"当前识别语言: {LANGUAGE}")

# 根据语言选择热词（英文或中文）
if LANGUAGE == "en":
    HOTWORDS = " ".join(COMMON_DRINKS)  # 英文饮料列表
else:
    # 中文饮料列表（请根据实际情况填写）
    HOTWORDS = "可乐 雪碧 芬达 美年达 七喜 果汁 橙汁 苹果汁 牛奶 酸奶 水 矿泉水 茶 红茶 绿茶 乌龙茶 咖啡 拿铁 卡布奇诺 啤酒 红酒"

print(f"热词列表: {HOTWORDS}")

# 加载SenseVoiceSmall模型
print("正在加载SenseVoiceSmall模型...")
model = AutoModel(
    model="iic/SenseVoiceSmall",
    trust_remote_code=True,
    vad_model="fsmn-vad",
    device="cuda"   # 如果有GPU则使用，否则改为 "cpu"
)
print("模型加载完成！")

app = FastAPI()

@app.post("/api/speech_recognition")
async def speech_recognition(audio: UploadFile = File(...)):
    """
    语音识别接口，支持热词增强和置信度过滤。
    """
    audio_data = await audio.read()
    
    # 根据 LANGUAGE 设置识别语言
    lang = "en" if LANGUAGE == "en" else "zh"
    
    # 调用模型，加入热词
    res = model.generate(
        input=audio_data,
        cache={},
        language=lang,
        use_itn=True,
        hotwords=HOTWORDS,          # 热词增强
        # 以下参数可能不受支持，如果报错请注释
        # hotwords_weight=2.0,       # 热词权重（仅部分版本支持）
        # return_spk_res=True,       # 返回置信度（仅部分版本支持）
    )
    
    if not res:
        return {"code": 200, "text": "识别失败"}
    
    # 尝试获取置信度（如果存在）
    # 不同版本的 FunASR 输出格式不同，常见为 res[0] 包含 'text' 和 'score'
    text = res[0].get("text", "").strip()
    score = res[0].get("score", 1.0)   # 默认置信度 1.0
    
    # 可选：过滤低置信度结果（阈值 0.3 可调整）
    # if score < 0.3:
    #     text = ""
    
    # 去除特殊标记（如 <|EN|> 等）
    text = text.replace("<|EN|>", "").replace("<|ZH|>", "").strip()
    
    return {"code": 200, "text": text}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
