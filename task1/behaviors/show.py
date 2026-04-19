from common.utils.show_camera import ColorViewer
from common.skills.camera import camera_manager as cams
from common.config import CAMERA_HEAD, CAMERA_CHEST

viewer = ColorViewer(cams.get(CAMERA_HEAD), cams.get(CAMERA_CHEST), window_names=["Head Camera", "Chest Camera"])
