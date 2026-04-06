"""
HDF5 Episode 录制模块
格式兼容 ALOHA / Soft-FOLD / ACT

使用示例:
    recorder = EpisodeRecorder("./data", freq=50)

    # 在 50Hz 主循环中:
    recorder.start_episode()
    for each tick:
        recorder.record_step(left_joints_deg, right_joints_deg,
                             left_gripper_open, right_gripper_open,
                             frames, timestamp)
    path = recorder.stop_episode()
"""

import os
import glob
import time
from concurrent.futures import ThreadPoolExecutor

import cv2
import h5py
import numpy as np


# 内部相机名 → HDF5 数据集名
CAM_NAME_MAP = {
    "cam_head": "cam_high",
    "cam_left_wrist": "cam_left_wrist",
    "cam_right_wrist": "cam_right_wrist",
}


def _encode_jpeg(bgr_frame: np.ndarray) -> bytes:
    """BGR → RGB → JPEG 编码，返回字节串。在线程池 worker 中调用。"""
    rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
    ok, buf = cv2.imencode('.jpg', rgb, [cv2.IMWRITE_JPEG_QUALITY, 95])
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    return buf.tobytes()


def save_episode_hdf5(
    path: str,
    qpos_list: list,
    timestamps: list,
    images: dict,
    freq: float,
    language_instruction: str,
) -> None:
    """将一个 episode 的缓冲数据写入 HDF5 文件。

    Parameters
    ----------
    path : 输出文件路径
    qpos_list : T 个 (14,) float32 数组的列表
    timestamps : T 个 float 的列表
    images : {cam_name: [T 个 bytes]} JPEG 编码后的图像
    freq : 采样频率 (Hz)
    language_instruction : 任务描述字符串
    """
    qpos = np.stack(qpos_list).astype(np.float32)  # (T, 14)
    T = qpos.shape[0]
    dt = 1.0 / freq

    # qvel: 有限差分，末帧为 0
    qvel = np.zeros_like(qpos)
    qvel[:-1] = (qpos[1:] - qpos[:-1]) / dt

    # action: qpos 前移一步，末帧复制
    action = np.zeros_like(qpos)
    action[:-1] = qpos[1:]
    action[-1] = qpos[-1]

    with h5py.File(path, 'w') as f:
        f.attrs['sim'] = False
        f.attrs['language_instruction'] = language_instruction

        f.create_dataset('observations/qpos', data=qpos)
        f.create_dataset('observations/qvel', data=qvel)
        f.create_dataset('action', data=action)
        f.create_dataset('time_stamp',
                         data=np.array(timestamps, dtype=np.float32))
        f.create_dataset('language_instruction',
                         data=language_instruction)

        # 图像: variable-length uint8 (JPEG bytes)
        dt_vlen = h5py.vlen_dtype(np.dtype('uint8'))
        for cam_name in ['cam_high', 'cam_left_wrist', 'cam_right_wrist']:
            jpeg_list = images.get(cam_name, [])
            dset = f.create_dataset(
                f'observations/images/{cam_name}',
                shape=(T,), dtype=dt_vlen,
            )
            for t, jpeg_bytes in enumerate(jpeg_list):
                dset[t] = np.frombuffer(jpeg_bytes, dtype=np.uint8)


class EpisodeRecorder:
    """HDF5 Episode 录制器。

    Parameters
    ----------
    out_dir : 输出目录（自动创建）
    freq : 采样频率，默认 50Hz
    language_instruction : 任务描述，写入 HDF5
    """

    def __init__(
        self,
        out_dir: str,
        freq: float = 50,
        language_instruction: str = "fold the cloth",
    ):
        self.out_dir = out_dir
        self.freq = freq
        self.language_instruction = language_instruction

        os.makedirs(out_dir, exist_ok=True)

        # 扫描已有 episode，从 max+1 开始编号
        existing = glob.glob(os.path.join(out_dir, "episode_*.hdf5"))
        if existing:
            indices = [
                int(os.path.basename(f).split("_")[1].split(".")[0])
                for f in existing
            ]
            self._episode_idx = max(indices) + 1
        else:
            self._episode_idx = 0

        self._is_recording = False
        self._jpeg_pool = ThreadPoolExecutor(
            max_workers=3, thread_name_prefix="jpeg"
        )
        self._reset_buffers()

    def _reset_buffers(self):
        self._buf_qpos = []
        self._buf_timestamps = []
        self._buf_images = {
            "cam_high": [],
            "cam_left_wrist": [],
            "cam_right_wrist": [],
        }

    @property
    def is_recording(self) -> bool:
        return self._is_recording

    def start_episode(self):
        """开始录制新 episode。"""
        self._reset_buffers()
        self._episode_start_time = time.time()
        self._is_recording = True

    def stop_episode(self) -> str:
        """停止录制，保存 HDF5 文件。返回保存路径，空 episode 返回空串。"""
        self._is_recording = False

        if not self._buf_qpos:
            return ""

        # resolve 所有 JPEG Future
        for cam_name in self._buf_images:
            self._buf_images[cam_name] = [
                f.result() if hasattr(f, 'result') else f
                for f in self._buf_images[cam_name]
            ]

        path = os.path.join(
            self.out_dir, f"episode_{self._episode_idx}.hdf5"
        )
        save_episode_hdf5(
            path=path,
            qpos_list=self._buf_qpos,
            timestamps=self._buf_timestamps,
            images=self._buf_images,
            freq=self.freq,
            language_instruction=self.language_instruction,
        )
        self._episode_idx += 1
        self._reset_buffers()
        return path

    def toggle(self) -> bool:
        """切换录制状态。返回 True=开始录制, False=停止并保存。"""
        if self._is_recording:
            duration = time.time() - self._episode_start_time
            frames = len(self._buf_qpos)
            path = self.stop_episode()
            if path:
                print(f"\n  [REC] 已保存: {path}  ({duration:.1f}s, {frames} 帧)")
            return False
        else:
            self.start_episode()
            return True

    def record_step(
        self,
        left_joints_deg,
        right_joints_deg,
        left_gripper_open,
        right_gripper_open,
        frames: dict,
        timestamp: float,
    ):
        """记录一帧数据。在 50Hz 主循环中调用。

        Parameters
        ----------
        left_joints_deg : 左臂 6 个关节角度 (度)
        right_joints_deg : 右臂 6 个关节角度 (度)
        left_gripper_open : 左夹爪状态 (0/1/None)
        right_gripper_open : 右夹爪状态 (0/1/None)
        frames : {内部相机名: BGR numpy array}
        timestamp : 时间戳 (秒)
        """
        # 度 → 弧度
        left_rad = np.radians(left_joints_deg, dtype=np.float32)
        right_rad = np.radians(right_joints_deg, dtype=np.float32)

        # 夹爪归一化: None 默认 1.0 (开)
        left_grip = float(left_gripper_open) if left_gripper_open is not None else 1.0
        right_grip = float(right_gripper_open) if right_gripper_open is not None else 1.0

        # 拼装 qpos (14,)
        qpos = np.array(
            [*left_rad, left_grip, *right_rad, right_grip],
            dtype=np.float32,
        )
        self._buf_qpos.append(qpos)
        self._buf_timestamps.append(timestamp)

        # JPEG 编码提交到线程池
        for internal_name, bgr_frame in frames.items():
            if bgr_frame is None:
                continue
            hdf5_name = CAM_NAME_MAP.get(internal_name)
            if hdf5_name is None:
                continue
            future = self._jpeg_pool.submit(_encode_jpeg, bgr_frame.copy())
            self._buf_images[hdf5_name].append(future)
