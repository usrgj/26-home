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
PROJECT_ROOT = Path(__file__).resolve().parents[2]
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

        view_station_id = _get_intro_view_station_id(second_guest.seat_id)

        _face_guest(view_station_id, second_guest.seat_id)
        _safe_speak(_build_intro_text(listener=second_guest, subject=first_guest, subject_index=0))

        _face_guest(view_station_id, first_guest.seat_id)
        _safe_speak(_build_intro_text(listener=first_guest, subject=second_guest, subject_index=1))

        return "receive_bag"


def _safe_speak(text: str) -> None:
    try:
        voice_assistant.speak(text)
    except Exception as exc:
        log.warning("介绍播报失败: %s", exc)


def _get_intro_view_station_id(fallback_seat_id: str) -> str:
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
    try:
        pan_tilt.home()
    except Exception as exc:
        log.warning("云台回中失败: %s", exc)

    if not view_station_id or not target_seat_id:
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
    station_angles = config.INTRO_LOOK_ANGLES_DEG.get(view_station_id, {})
    angle_deg = station_angles.get(target_seat_id)
    return float(angle_deg) if angle_deg is not None else None


def _get_seat_nav_id(seat_id: str) -> str:
    for seat_mapping in config.SEATS_MAPPING:
        if seat_mapping["seat_id"] == seat_id:
            return str(seat_mapping["nav_id"])
    return ""


def _build_intro_text(listener, subject, subject_index: int) -> str:
    """生成英文介绍文案"""
    listener_name = (getattr(listener, "name", "") or "").strip()
    subject_name = (getattr(subject, "name", "") or "").strip()
    favorite_drink = (getattr(subject, "favorite_drink", "") or "").strip()
    subject_features = getattr(subject, "visual_features", {}) or {}

    # 英文始终使用英文回退名称
    fallback_names = ["the first guest", "the second guest"]
    subject_label = subject_name or fallback_names[subject_index]
    listener_label = listener_name or "this guest"

    sentences = [f"{listener_label}, this is {subject_label}."]
    if favorite_drink:
        sentences.append(f"{subject_label}'s favorite drink is {favorite_drink}.")
    feature_phrase = _build_visual_feature_phrase_en(subject_features)
    if feature_phrase:
        sentences.append(f"{subject_label} {feature_phrase}.")
    return " ".join(sentences)


def _build_visual_feature_phrase_en(features: dict) -> str:
    """英文视觉特征描述，返回以动词开头的短语（不含主语）"""
    if not features:
        return ""

    parts = []
    # 性别
    gender = features.get("gender", "")
    if gender:
        gender_word = "a man" if "male" in str(gender).lower() else "a woman" if "female" in str(gender).lower() else ""
        if gender_word:
            parts.append(f"is {gender_word}")
    # 头发颜色
    hair_color = features.get("hair_color") or features.get("hair color")
    if hair_color:
        parts.append(f"has {hair_color} hair")
    # 衣服颜色
    clothing_color = features.get("clothing_color") or features.get("clothes color")
    if clothing_color:
        parts.append(f"is wearing {clothing_color} clothes")
    # 眼镜
    glasses = features.get("glasses")
    if glasses and str(glasses).lower() not in ["none", "no", "false", ""]:
        parts.append("wears glasses")
    # 帽子
    hat = features.get("hat")
    if hat and str(hat).lower() not in ["none", "no", "false", ""]:
        parts.append("wears a hat")

    if not parts:
        return ""

    # 将第一个部分作为主谓结构，后续用逗号连接
    if parts[0].startswith("is "):
        # 比如 "is a man" 后面接 "wears glasses" 需要调整
        # 更自然的做法：全部用第三人称单数动词
        # 简单处理：将第一个部分保留，其余部分加上 "and" 或逗号
        if len(parts) == 1:
            return parts[0]
        else:
            # 将第一个部分（如 "is a man"）替换为 "is a man who" 然后连接
            rest = ", ".join(parts[1:])
            return parts[0] + " who " + rest
    else:
        # 如果没有 is 开头，直接用逗号连接
        return ", ".join(parts)

# 注意：本文件已移除所有中文分支，仅保留英文。若需要中文请恢复相应函数。
