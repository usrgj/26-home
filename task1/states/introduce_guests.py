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

import logging
import math
import sys
from pathlib import Path

from common.state_machine import State
from common.skills.agv_api import agv, wait_nav
from common.skills.audio_module.voice_assiant import voice_assistant
from common.skills.head_control import pan_tilt
from task1 import config

# 导入语言配置
PROJECT_ROOT = Path(__file__).resolve().parents[2]  # 向上2级到 26-home
sys.path.insert(0, str(PROJECT_ROOT))
from common.config import LANGUAGE

log = logging.getLogger("task1.introduce_guests")


class IntroduceGuests(State):
    def execute(self, ctx) -> str:
        guests = ctx.guests
        if len(guests) < 2:
            log.warning("客人数量不足，跳过介绍阶段")
            return "receive_bag"

        first_guest = guests[0]
        second_guest = guests[1]

        # 状态1结束后通常停在第二位客人的导航站点附近；若取不到则退化为第二位客人的 seat_id 对应站点。
        view_station_id = _get_intro_view_station_id(second_guest.seat_id)

        _face_guest(view_station_id, second_guest.seat_id)
        _safe_speak(_build_intro_text(listener=second_guest, subject=first_guest, subject_index=0))

        _face_guest(view_station_id, first_guest.seat_id)
        _safe_speak(_build_intro_text(listener=first_guest, subject=second_guest, subject_index=1))

        return "receive_bag"


def _safe_speak(text: str) -> None:
    """尽量播报介绍语句，失败时不打断主流程。"""
    try:
        voice_assistant.speak(text)
    except Exception as exc:
        log.warning("介绍播报失败: %s", exc)


def _get_intro_view_station_id(fallback_seat_id: str) -> str:
    """确定介绍阶段当前所处的观察站点。"""
    current_station = agv.get_current_station() or ""
    if current_station:
        return current_station

    fallback_nav_id = _get_seat_nav_id(fallback_seat_id)
    if fallback_nav_id:
        log.warning("未获取到当前导航站点，退化使用 seat_id=%s 对应站点 %s", fallback_seat_id, fallback_nav_id)
        return fallback_nav_id

    log.warning("未获取到当前导航站点，且无法从 seat_id=%s 推导站点", fallback_seat_id)
    return ""


def _face_guest(view_station_id: str, target_seat_id: str) -> None:
    """通过底盘原地转向，让机器人面向目标客人。"""
    try:
        pan_tilt.home()
    except Exception as exc:
        log.warning("云台回中失败: %s", exc)

    if not view_station_id:
        log.warning("缺少介绍观察站点，跳过底盘转向")
        return

    if not target_seat_id:
        log.warning("目标客人缺少 seat_id，跳过底盘转向")
        return

    angle_deg = _get_guest_facing_angle_deg(view_station_id, target_seat_id)
    if angle_deg is None:
        log.warning("未配置介绍角度: view_station=%s, target_seat=%s", view_station_id, target_seat_id)
        return

    pose = agv.get_pose()
    if pose is None:
        log.warning("无法获取机器人位姿，跳过底盘转向")
        return

    try:
        x = float(pose["x"])
        y = float(pose["y"])
    except (KeyError, TypeError, ValueError) as exc:
        log.warning("机器人位姿不完整，跳过底盘转向: %s", exc)
        return

    theta = math.radians(angle_deg)
    ok = agv.free_navigate_to(x, y, theta)
    if not ok:
        log.warning("原地转向指令发送失败: view_station=%s target_seat=%s", view_station_id, target_seat_id)
        return

    wait_nav(timeout=config.NAV_TIMEOUT)


def _get_guest_facing_angle_deg(view_station_id: str, target_seat_id: str) -> float | None:
    """从介绍阶段角度配置里查询目标朝向角度（度）。"""
    station_angles = config.INTRO_LOOK_ANGLES_DEG.get(view_station_id, {})
    angle_deg = station_angles.get(target_seat_id)
    if angle_deg is None:
        return None
    return float(angle_deg)


def _get_seat_nav_id(seat_id: str) -> str:
    """从 seat_id 查其对应的导航站点。"""
    for seat_mapping in config.SEATS_MAPPING:
        if seat_mapping["seat_id"] == seat_id:
            return str(seat_mapping["nav_id"])
    return ""


def _build_intro_text(listener, subject, subject_index: int) -> str:
    """根据全局语言配置生成介绍文案（英文或中文）。"""
    listener_name = (getattr(listener, "name", "") or "").strip()
    subject_name = (getattr(subject, "name", "") or "").strip()
    favorite_drink = (getattr(subject, "favorite_drink", "") or "").strip()
    subject_features = getattr(subject, "visual_features", {}) or {}

    if LANGUAGE == "en":
        listener_label = listener_name or "this guest"
        subject_label = subject_name or _get_guest_fallback_name_en(subject_index)
        sentences = [f"{listener_label}, this is {subject_label}."]
        if favorite_drink:
            sentences.append(f"{subject_label}'s favorite drink is {favorite_drink}.")
        feature_phrase = _build_visual_feature_phrase_en(subject_features)
        if feature_phrase:
            sentences.append(f"{subject_label} {feature_phrase}.")
        return " ".join(sentences)
    else:
        # 中文版本
        listener_label = listener_name or "这位客人"
        subject_label = subject_name or _get_guest_fallback_name_zh(subject_index)
        sentences = [f"{listener_label}，这位是{subject_label}。"]
        if favorite_drink:
            # 假设 favorite_drink 是英文，可以保留或简单处理
            sentences.append(f"{subject_label}最喜欢的饮料是{favorite_drink}。")
        feature_phrase = _build_visual_feature_phrase_zh(subject_features)
        if feature_phrase:
            sentences.append(f"{subject_label}{feature_phrase}。")
        return "".join(sentences)


def _get_guest_fallback_name_en(subject_index: int) -> str:
    return "the first guest" if subject_index == 0 else "the second guest"


def _get_guest_fallback_name_zh(subject_index: int) -> str:
    return "第一位客人" if subject_index == 0 else "第二位客人"


def _build_visual_feature_phrase_en(features: dict) -> str:
    """英文视觉特征描述。"""
    if not features:
        return ""
    parts = []
    gender = str(features.get("gender", "")).strip()
    if gender:
        gender_en = "male" if "man" in gender else "female" if "female" in gender else gender
        parts.append(gender_en)
    hair_color = str(features.get("hair color", "")).strip()
    if hair_color:
        parts.append(f"has {hair_color} hair")
    clothes_color = str(features.get("clothes color", "")).strip()
    if clothes_color:
        parts.append(f"wears {clothes_color} clothes")
    glasses = str(features.get("glasses", "")).strip()
    if glasses and "wears" in glasses.lower():
        parts.append("wears glasses")
    hat = str(features.get("hat", "")).strip()
    if hat and "wear" in hat.lower() and "not" not in hat.lower():
        parts.append("wears a hat")
    if not parts:
        return ""
    return "who is " + ", ".join(parts)


def _build_visual_feature_phrase_zh(features: dict) -> str:
    """中文视觉特征描述（简单映射）。"""
    if not features:
        return ""
    parts = []
    gender = str(features.get("gender", "")).strip()
    if gender:
        if "male" in gender.lower():
            parts.append("男性")
        elif "female" in gender.lower():
            parts.append("女性")
        else:
            parts.append(gender)
    hair_color = str(features.get("hair color", "")).strip()
    if hair_color:
        parts.append(f"{hair_color}头发")
    clothes_color = str(features.get("clothes color", "")).strip()
    if clothes_color:
        parts.append(f"穿着{clothes_color}衣服")
    glasses = str(features.get("glasses", "")).strip()
    if glasses and "wears" in glasses.lower():
        parts.append("戴眼镜")
    hat = str(features.get("hat", "")).strip()
    if hat and "wear" in hat.lower() and "not" not in hat.lower():
        parts.append("戴帽子")
    if not parts:
        return ""
    return "，".join(parts)