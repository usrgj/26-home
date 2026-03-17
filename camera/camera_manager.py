# camera_manager.py
import pyrealsense2 as rs
import numpy as np
from typing import Optional


class RealSenseCamera:
    """单个相机的封装"""

    def __init__(self, serial: str):
        self.serial = serial
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.align = None
        self.started = False

    def start(self, width=640, height=480, fps=30):
        if self.started:
            return
        self.config.enable_device(self.serial)
        self.config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        self.config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        self.pipeline.start(self.config)
        self.align = rs.align(rs.stream.color)
        self.started = True

    def get_frames(self):
        if not self.started:
            raise RuntimeError(f"相机 {self.serial} 未启动")
        frames = self.pipeline.wait_for_frames()
        aligned = self.align.process(frames)
        color_frame = aligned.get_color_frame()
        depth_frame = aligned.get_depth_frame()
        if not color_frame or not depth_frame:
            return None, None
        return np.asanyarray(color_frame.get_data()), np.asanyarray(depth_frame.get_data())

    def stop(self):
        if self.started:
            self.pipeline.stop()
            self.started = False


class CameraManager:
    """相机管理器：按需获取、懒启动、统一管理生命周期"""

    def __init__(self):
        self._cameras: dict[str, RealSenseCamera] = {}

    @staticmethod
    def list_devices() -> list[str]:
        """列出所有已连接相机的序列号"""
        ctx = rs.context()
        return [d.get_info(rs.camera_info.serial_number) for d in ctx.devices]

    def get(self, serial: str) -> RealSenseCamera:
        """获取相机对象，不存在则创建（但不自动启动）"""
        if serial not in self._cameras:
            self._cameras[serial] = RealSenseCamera(serial)
        return self._cameras[serial]

    def start(self, serial: str, **kwargs) -> RealSenseCamera:
        """获取并启动相机，一步到位"""
        cam = self.get(serial)
        cam.start(**kwargs)
        return cam

    def stop(self, serial: str):
        """停止指定相机"""
        if serial in self._cameras:
            self._cameras[serial].stop()

    def stop_all(self):
        """停止所有相机"""
        for cam in self._cameras.values():
            cam.stop()


# 管理器本身是模块级单例
camera_manager = CameraManager()