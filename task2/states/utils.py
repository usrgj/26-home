"""task2 状态共享的小工具函数。"""

from __future__ import annotations

import logging

log = logging.getLogger("task2.states")


def safe_speak(text: str) -> None:
    """尽量播报文本，语音服务失败时不打断状态机。"""
    try:
        from common.skills.audio_module.voice_assiant import voice_assistant

        voice_assistant.speak(text)
    except Exception as exc:
        log.warning("语音播报失败: %s", exc)


def camera_serial_for_role(role: str) -> str:
    """把 task2 配置中的相机角色映射到全局相机序列号。"""
    from common.config import CAMERA_CHEST, CAMERA_HEAD

    mapping = {
        "head": CAMERA_HEAD,
        "chest": CAMERA_CHEST,
        # "left": CAMERA_LEFT,
        # "right": CAMERA_RIGHT,
    }
    if role not in mapping:
        raise KeyError(f"未知相机角色: {role}")
    return mapping[role]


def navigate_to_station(target: str, timeout: float) -> bool:
    """导航到指定站点，并等待导航完成。"""
    from common.skills.agv_api import agv, wait_nav

    if not target:
        log.warning("目标站点为空，跳过导航")
        return False

    source = agv.get_current_station() or ""
    result = agv.navigate_to(source, target)
    if result is None:
        log.warning("导航指令发送失败: source=%s target=%s", source, target)
        return False

    return wait_nav(timeout=timeout)
