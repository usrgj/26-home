"""ReID 画面来源适配层。

作用：
    为独立 OpenCV 测试和后续 task 接入提供统一的 read()/get_frame() 接口。
    OpenCVCameraSource 直接打开本机相机；CameraManagerFrameSource 读取
    common.skills.camera.camera_manager 管理的缓存帧。

用法：
    source = OpenCVCameraSource(index=6).start()
    frame = source.read()
    source.stop()

    source = CameraManagerFrameSource(camera_manager.get(CAMERA_CHEST))
    frame = source.read()
"""

from __future__ import annotations

from typing import Any


def _load_cv2():
    """Import OpenCV only when a local camera is opened."""

    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError(
            "Missing OpenCV dependency. Install opencv-python before opening "
            "an OpenCV camera source."
        ) from exc

    return cv2


class OpenCVCameraSource:
    """Open one local camera with cv2.VideoCapture.

    Usage:
        source = OpenCVCameraSource(index=6).start()
        frame = source.read()
        source.stop()
    """

    def __init__(
        self,
        index: int = 6,
        width: int = 1280,
        height: int = 720,
        fps: int = 30,
        backend: int | None = None,
    ) -> None:
        """Store OpenCV camera parameters without opening the device."""

        self.index = int(index)
        self.width = int(width)
        self.height = int(height)
        self.fps = int(fps)
        self.backend = backend
        self._cv2 = None
        self._cap = None

    def start(self) -> "OpenCVCameraSource":
        """Open the OpenCV camera if it is not already open."""

        if self._cap is not None:
            return self

        cv2 = _load_cv2()
        if self.backend is None:
            cap = cv2.VideoCapture(self.index)
        else:
            cap = cv2.VideoCapture(self.index, self.backend)

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        cap.set(cv2.CAP_PROP_FPS, self.fps)

        if not cap.isOpened():
            cap.release()
            raise RuntimeError(f"Failed to open OpenCV camera index {self.index}")

        self._cv2 = cv2
        self._cap = cap
        return self

    def read(self):
        """Read one BGR frame, returning None when capture fails."""

        if self._cap is None:
            self.start()

        ok, frame = self._cap.read()
        if not ok:
            return None

        return frame

    def get_frame(self):
        """Return one BGR frame for callers that prefer get_frame naming."""

        return self.read()

    def stop(self) -> None:
        """Release the OpenCV camera device."""

        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __enter__(self) -> "OpenCVCameraSource":
        """Open the source for use in a with block."""

        return self.start()

    def __exit__(self, exc_type, exc, tb) -> None:
        """Release the source when leaving a with block."""

        self.stop()


class CameraManagerFrameSource:
    """Read color frames from common.skills.camera.camera_manager cameras.

    Usage:
        source = CameraManagerFrameSource(camera_manager.get(CAMERA_CHEST))
        frame = source.read()
    """

    def __init__(self, camera: Any) -> None:
        """Store a camera object that exposes get_frames()."""

        self.camera = camera

    def start(self) -> "CameraManagerFrameSource":
        """Return self because camera_manager owns the camera lifecycle."""

        return self

    def read(self):
        """Return the latest color frame from camera.get_frames()."""

        frames = self.camera.get_frames()
        if frames is None:
            return None

        if isinstance(frames, (tuple, list)):
            return frames[0] if frames else None

        return frames

    def get_frame(self):
        """Return one color frame for callers that prefer get_frame naming."""

        return self.read()

    def stop(self) -> None:
        """Leave lifecycle management to camera_manager."""

        return None
