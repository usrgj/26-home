from common.utils.show_camera import ColorViewer
from common.skills.camera import camera_manager as cams
from common.config import CAMERA_HEAD, CAMERA_CHEST

head_viewer = ColorViewer(cams.get(CAMERA_HEAD))
chest_viewer = ColorViewer(cams.get(CAMERA_CHEST))