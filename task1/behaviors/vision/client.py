from openai import OpenAI
import base64
import json
import cv2
import tempfile
import os
import re  




prompt = ('Please only output the person\'s appearance features in the following JSON format (in English): '
'{"hair_color": "string or None", "hat": "True | False | None", "glasses": "True | False | None", "clothing_color": "string or None", "gender": "man | lady | None"} '
'For example: {"hair_color": "black", "hat": "False", "glasses": "True", "clothing_color": "white", "gender": "man"}. '
'Do NOT add any explanation or extra words.'
    )

client = OpenAI(
    api_key="retoo",
    base_url="http://127.0.0.1:8003/v1"
)

def image_to_base64(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

# ===================== 新增：清理思考内容，提取JSON =====================
def clean_response_text(text):
    # 提取第一个大括号里的JSON（最通用）
    match = re.search(r'\{[\s\S]*?\}', text)
    if match:
        return match.group(0)
    # 如果没找到，尝试清理其它包裹
    text = re.sub(r'```json\n', '', text)
    text = re.sub(r'```', '', text)
    return text.strip()

# ======================================================================

def analyze_person_features(image_path):
    """调用大模型分析外貌特征"""
    try:
        print("🔄 大模型推理中...")
        
        messages = [
                    {
            "role": "system",
            "content": "你是一个只输出合法JSON的视觉特征提取助手。"
        },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "data:image/jpeg;base64," + image_to_base64(image_path)
                        }
                    },
                    {
                        "type": "text",
                        "text": (
                            prompt
                        )
                    }
                ]
            }
        ]
        
        chat_response = client.chat.completions.create(
            model="Qwen3.5-4B",
            messages=messages,
            max_tokens=4096,
            temperature=0.6,
            top_p=0.8,
            presence_penalty=0.5,
            extra_body={
                "top_k": 20,
                "chat_template_kwargs": {"enable_thinking": False},
            }, 
        )
        
        text_result = chat_response.choices[0].message.content
        print(f"原始响应: {text_result}")

        # ===================== 在这里加入清理 =====================
        text_result = clean_response_text(text_result)
        print(f"清理后: {text_result}")
        # ==========================================================

        struct_result = json.loads(text_result)
        print("✓ 推理完成")
        
        if not isinstance(struct_result, dict): 
           struct_result = {"描述": str(struct_result)}
        return struct_result
        
    except json.JSONDecodeError as e:
        print(f"❗ JSON解析失败: {e}")
        print(f"   最终尝试解析内容: {text_result}") 
        return {}
    except Exception as e:
        print(f"❗ 大模型推理失败: {e}")
        return {}
