from openai import OpenAI
import base64
import json
import cv2
import tempfile
import os

client = OpenAI(
    api_key="retoo",
    base_url="http://127.0.0.1:8004/v1"
)

def image_to_base64(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def analyze_person_features(image_path):
    """调用大模型分析外貌特征"""
    try:
        print("🔄 大模型推理中...")
        
        messages = [
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
                            "请仔细分析此人的外貌特征，只用JSON格式输出，字段为：头发颜色、是否戴帽子、是否戴眼镜、衣服颜色、性别。"
                            ' 例如：{"头发颜色": "黑色", "帽子": "未戴", "眼镜": "佩戴", "衣服颜色": "白色", "性别": "男性"}'
                        )
                    }
                ]
            }
        ]
        
        chat_response = client.chat.completions.create(
            model="Qwen3.5-4B",
            messages=messages,
            max_tokens=4096,
            temperature=0.7,
            top_p=0.8,
            presence_penalty=1.5,
            extra_body={
                "top_k": 20,
                "chat_template_kwargs": {"enable_thinking": False},  # ★ 关闭深度思考
            }, 
        )
        
        text_result = chat_response.choices[0].message.content
        print(f"原始响应: {text_result}")  # ★ 打印原始响应
        struct_result = json.loads(text_result)
        print("✓ 推理完成")
        if not isinstance(struct_result, dict): 
           struct_result = {"描述": str(struct_result)}
        return struct_result
        
    except json.JSONDecodeError as e:
        print(f"❗ JSON解析失败: {e}")
        print(f"   原始内容: {text_result}") 
        return {}
    except Exception as e:
        print(f"❗ 大模型推理失败: {e}")
        return {}


