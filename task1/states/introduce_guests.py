"""
状态2：相互介绍两位客人

评分项:
  - 准确说出每位客人的姓名及喜爱饮品
  - 提及一位客人时需注视另一位对应客人

流程:
  1. 先面向第二位客人，介绍第一位客人
  2. 再原地转向第一位客人，介绍第二位客人
"""
from __future__ import annotations

import math

from common.state_machine import State
from common.skills.agv_api import agv, wait_nav
from common.skills.audio_module.voice_assiant import voice_assistant
from common.skills.head_control import pan_tilt
from task1 import config

# 导入语言配置
from common.config import LANGUAGE


class IntroduceGuests(State):
    def execute(self, ctx) -> str:
        guests = ctx.guests
        if len(guests) < 2:
            return "receive_bag"

        first_guest = guests[0]
        second_guest = guests[1]

        # 状态1结束后通常停在第二位客人的导航站点附近，优先使用这个站点的朝向配置
        current_station = agv.get_current_station()

        face_to_guest(current_station, second_guest.seat_id) # 面向第二位客人 
        _safe_speak(_build_intro_text(listener=second_guest, subject=first_guest, subject_index=0))

        face_to_guest(current_station, first_guest.seat_id)
        _safe_speak(_build_intro_text(listener=first_guest, subject=second_guest, subject_index=1))

        return "receive_bag"



# 转向相关
def face_to_guest(view_station_id: str, target_seat_id: str) -> None:
    """
    通过底盘原地转向，让机器人面向目标客人。
    target_seat_id: 目标客人所在座位 ID
    """
    try:
        pan_tilt.home()
    except Exception:
        pass

    if not view_station_id:
        return

    if not target_seat_id:
        return

    angle_deg = _get_guest_facing_angle_deg(view_station_id, target_seat_id)
    if angle_deg is None:
        return

    theta = math.radians(angle_deg)
    ok = agv.navigate_to(agv.get_current_station(), view_station_id, angle=theta)
    if not ok:
        return

    wait_nav(timeout=config.NAV_TIMEOUT)


def _get_guest_facing_angle_deg(view_station_id: str, target_seat_id: str) -> float | None:
    """从介绍阶段角度配置里查询目标朝向角度（度）。"""
    print("当前站点：%s，目标座位：%s" % (view_station_id, target_seat_id), end="")
    station_angles = config.INTRO_LOOK_ANGLES_DEG.get(view_station_id, {})
    angle_deg = station_angles.get(target_seat_id)
    print("将转至角度：%s" % angle_deg)
    if angle_deg is None:
        return None
    return float(angle_deg)



# 介绍相关
def _safe_speak(text: str) -> None:
    """尽量播报介绍语句，失败时不打断主流程。"""
    try:
        voice_assistant.speak(text)
    except Exception:
        pass


def _build_intro_text(listener, subject, subject_index: int) -> str:
    """根据全局语言配置生成介绍文案（英文或中文）。"""
    listener_name = (getattr(listener, "name", "") or "").strip()
    subject_name = (getattr(subject, "name", "") or "").strip()
    favorite_drink = (getattr(subject, "favorite_drink", "") or "").strip()
    subject_features = getattr(subject, "visual_features", {}) or {}
    is_en = LANGUAGE == "en"

    fallback_names = {
        True: ("the first guest", "the second guest"),
        False: ("第一位客人", "第二位客人"),
    }
    subject_fallbacks = fallback_names[is_en]
    
    subject_fallback = subject_fallbacks[0] if subject_index == 0 else subject_fallbacks[1] # index为0时介绍第一位客人，index为1时介绍第二位客人

    listener_label = listener_name or ("this guest" if is_en else "这位客人")
    subject_label = subject_name or subject_fallback
    
    # 构建视觉特征描述文案，因为只需要描述一位客人，所以只当index为0时生效
    feature_phrase = (
        build_visual_feature_phrase(subject_features)
        if subject_index == 0 else ""
    )

    if is_en:
        # 英文介绍
        text = f"{listener_label}, that is {subject_label}."
        if favorite_drink:
            text += f" {subject_label}'s favorite drink is {favorite_drink}."
        if feature_phrase:
            # feature_phrase 已包含主语（如 "He is wearing a hat and glasses"）
            text += f" {feature_phrase}"
        return text
    
    else:
        text = f"{listener_label}，"
        if feature_phrase:
            text += feature_phrase
        text += f"他是{subject_label}。"
        if favorite_drink:
            text += f"他最喜欢的饮料是{favorite_drink}。"

        return text

def build_visual_feature_phrase(features: dict) -> str:
    '''
    构建视觉描述文本（英文）
    '''
    result = ""
    # 不定特征
    hair_color = features.get("hair_color")
    clothing_color = features.get("clothing_color")
    
    # 二元特征
    gender = features.get("gender")
    
    # 布尔特征
    hat = features.get("hat")
    glasses = features.get("glasses")
    
    if LANGUAGE == "en":
        # 英文描述，例如 "He is wearing a hat and glasses, with black hair and a red shirt."
        # 注意：为了自然，我们构建一个以主语开头的句子（主语由调用方补充）
        # 这里返回的短语应该可以直接接在主语后面，例如 " is wearing a hat..."
        # 但调用方会写成 "{subject_label} {feature_phrase}"，所以 feature_phrase 应以动词或形容词开头
        # 简单起见，返回一个完整的小句（不带主语），由调用方拼接
        parts = []
        if hat and glasses:
            parts.append("wearing a hat and glasses")
        elif hat:
            parts.append("wearing a hat")
        elif glasses:
            parts.append("wearing glasses")
        
        if hair_color:
            parts.append(f"has {hair_color} hair")
        if clothing_color:
            parts.append(f"is wearing {clothing_color} clothes")
        if gender:
            # gender 可能是 "male" 或 "female"，可转换为 "a man" / "a woman"
            gender_word = "a man" if gender.lower() == "male" else "a woman" if gender.lower() == "female" else gender
            parts.insert(0, f"is {gender_word}")
        
        if parts:
            # 组合成 "is a man, wearing a hat and glasses, has black hair"
            # 但更好的语法：第一个部分如果是 "is a man" 则不加逗号，后续用逗号
            result = ", ".join(parts)
            # 确保句子开头大写（调用方会处理）
        else:
            result = ""
        return result
    else:
        result += "那边那位"
            
        if hat:
            result += "戴着帽子，"
        else:
            result += "没戴帽子，"
        
        if glasses:
            result += "戴着眼镜，"
        else:
            result += "没戴眼镜，"
            
        if hair_color:
            result += "%s头发，" % hair_color
            
        if clothing_color:
            result += "穿着%s衣服的" % clothing_color
        
        if gender:
            result += gender
            
        result += "客人，"
        return result
