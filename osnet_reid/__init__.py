"""osnet_reid 包公共入口。

作用：
    统一导出 OSNet ReID 的稳定接口，供独立测试脚本和后续 task1 接入使用。
    导入本包不会打开相机、加载 YOLO、加载 OSNet 或初始化机器人硬件。

用法：
    from osnet_reid import OSNetReID, OpenCVCameraSource
    from osnet_reid import CameraManagerFrameSource, FeatureGallery
"""

from .core import DEFAULT_MODEL_PATH, OSNetReID
from .gallery import DEFAULT_GALLERY_PATH, FeatureGallery, GalleryIdentity
from .matcher import IdentityMatcher, MatchScore
from .sources import CameraManagerFrameSource, OpenCVCameraSource

__all__ = [
    "DEFAULT_MODEL_PATH",
    "DEFAULT_GALLERY_PATH",
    "OSNetReID",
    "FeatureGallery",
    "GalleryIdentity",
    "IdentityMatcher",
    "MatchScore",
    "OpenCVCameraSource",
    "CameraManagerFrameSource",
]
