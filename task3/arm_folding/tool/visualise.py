"""Offline episode visualizer for arm_folding data.

Examples:
    python3 task3/arm_folding/tool/visualise.py
    python3 task3/arm_folding/tool/visualise.py --episode 3
    python3 task3/arm_folding/tool/visualise.py --file task3/arm_folding/data/episode_3.hdf5
    python3 task3/arm_folding/tool/visualise.py --episode 3 --save-video
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path

try:
    import numpy as np
except ImportError:  # pragma: no cover - runtime environment dependent
    np = None

try:
    import cv2
except ImportError:  # pragma: no cover - runtime environment dependent
    cv2 = None

try:
    import h5py
except ImportError:  # pragma: no cover - runtime environment dependent
    h5py = None


ARM_FOLDING_DIR = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = ARM_FOLDING_DIR / "data"
CAMERA_ORDER = ("cam_high", "cam_left_wrist", "cam_right_wrist")
CAMERA_TITLES = {
    "cam_high": "Head",
    "cam_left_wrist": "Left Wrist",
    "cam_right_wrist": "Right Wrist",
}
DEFAULT_FPS = 50.0
QPOS_DIM = 14
GRIPPER_OPEN_THRESHOLD = 0.5
EPISODE_RE = re.compile(r"episode_(\d+)\.hdf5$")
WINDOW_NAME = "arm_folding episode visualiser"


@dataclass(frozen=True)
class EpisodeData:
    path: Path
    qpos: np.ndarray
    timestamps: np.ndarray
    images: dict[str, list[object]]
    num_frames: int


@dataclass(frozen=True)
class Layout:
    target_height: int
    camera_widths: tuple[int, ...]
    top_width: int
    panel_height: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualise one arm_folding episode.")
    parser.add_argument("--episode", type=int, help="Load episode_<n>.hdf5 from data dir.")
    parser.add_argument("--file", type=Path, help="Load a specific .hdf5 file.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help=f"Dataset directory. Default: {DEFAULT_DATA_DIR}",
    )
    parser.add_argument(
        "--fps",
        type=float,
        help="Playback/export FPS. If omitted, infer from timestamps and fallback to 50.",
    )
    parser.add_argument(
        "--save-video",
        action="store_true",
        help="Export a visualized MP4 alongside interactive playback.",
    )
    parser.add_argument("--out", type=Path, help="Output path for exported MP4.")
    parser.add_argument(
        "--start-frame",
        type=int,
        default=0,
        help="Initial frame index for interactive playback.",
    )
    args = parser.parse_args()
    if args.out is not None and not args.save_video:
        parser.error("--out requires --save-video")
    if args.fps is not None and args.fps <= 0:
        parser.error("--fps must be positive")
    if args.start_frame < 0:
        parser.error("--start-frame must be >= 0")
    return args


def ensure_runtime_dependencies() -> None:
    missing = []
    if np is None:
        missing.append("numpy")
    if cv2 is None:
        missing.append("opencv-python")
    if h5py is None:
        missing.append("h5py")
    if missing:
        raise RuntimeError(f"Missing dependencies: {', '.join(missing)}")


def resolve_episode_path(data_dir: Path, episode: int | None, file_path: Path | None) -> Path:
    if file_path is not None:
        path = file_path.expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Episode file not found: {path}")
        return path

    data_dir = data_dir.expanduser().resolve()
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    candidates = []
    for path in sorted(data_dir.glob("episode_*.hdf5")):
        match = EPISODE_RE.fullmatch(path.name)
        if match:
            candidates.append((int(match.group(1)), path))

    if not candidates:
        raise FileNotFoundError(f"No episode_*.hdf5 found in {data_dir}")

    if episode is not None:
        for idx, path in candidates:
            if idx == episode:
                return path
        raise FileNotFoundError(f"episode_{episode}.hdf5 not found in {data_dir}")

    return max(candidates, key=lambda item: item[0])[1]


def require_dataset(root: "h5py.File", path: str):
    dataset = root.get(path)
    if dataset is None:
        raise KeyError(f"Required dataset missing: {path}")
    return dataset


def load_episode(path: Path) -> EpisodeData:
    with h5py.File(path, "r") as root:
        qpos = np.asarray(require_dataset(root, "observations/qpos")[()], dtype=np.float32)
        timestamps = np.asarray(require_dataset(root, "time_stamp")[()], dtype=np.float64).reshape(-1)

        if qpos.ndim != 2 or qpos.shape[1] != QPOS_DIM:
            raise ValueError(f"observations/qpos must have shape (T, {QPOS_DIM}), got {qpos.shape}")

        images: dict[str, list[object]] = {}
        lengths = [len(qpos), len(timestamps)]
        for cam_name in CAMERA_ORDER:
            dataset = require_dataset(root, f"observations/images/{cam_name}")
            frames = list(dataset[()])
            images[cam_name] = frames
            lengths.append(len(frames))

    num_frames = min(lengths)
    if num_frames <= 0:
        raise ValueError(f"Episode contains no valid frames: {path}")

    if len(set(lengths)) != 1:
        print(
            f"Warning: inconsistent lengths detected {lengths}, truncating to {num_frames} frames.",
            file=sys.stderr,
        )

    return EpisodeData(
        path=path,
        qpos=qpos[:num_frames],
        timestamps=timestamps[:num_frames],
        images={cam_name: frames[:num_frames] for cam_name, frames in images.items()},
        num_frames=num_frames,
    )


def decode_image(raw: object, cam_name: str, frame_idx: int) -> np.ndarray:
    if isinstance(raw, np.ndarray) and raw.ndim == 3:
        if raw.dtype != np.uint8:
            return raw.astype(np.uint8)
        return raw.copy()

    if isinstance(raw, np.ndarray) and raw.ndim == 1 and raw.dtype == np.uint8:
        buf = raw
    elif isinstance(raw, (bytes, bytearray, memoryview)):
        buf = np.frombuffer(raw, dtype=np.uint8)
    else:
        arr = np.asarray(raw)
        if arr.ndim == 1 and arr.dtype == np.uint8:
            buf = arr
        else:
            raise ValueError(f"Unsupported image payload for {cam_name} frame {frame_idx}")

    image = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Failed to decode JPEG for {cam_name} frame {frame_idx}")

    # collect_dataset.py encodes BGR capture frames after swapping to RGB,
    # so swap back after imdecode to display the original colors in OpenCV.
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def infer_fps(timestamps: np.ndarray, fps_override: float | None) -> float:
    if fps_override is not None:
        return fps_override

    if timestamps.size < 2:
        return DEFAULT_FPS

    deltas = np.diff(timestamps)
    valid = deltas[np.isfinite(deltas) & (deltas > 1e-6)]
    if valid.size == 0:
        return DEFAULT_FPS
    return float(1.0 / np.median(valid))


def compute_frame_delay_ms(frame_idx: int, timestamps: np.ndarray, fps_override: float | None) -> int:
    if fps_override is not None:
        return max(1, int(round(1000.0 / fps_override)))

    if frame_idx >= len(timestamps) - 1:
        return max(1, int(round(1000.0 / DEFAULT_FPS)))

    delta = timestamps[frame_idx + 1] - timestamps[frame_idx]
    if not np.isfinite(delta) or delta <= 1e-6:
        return max(1, int(round(1000.0 / DEFAULT_FPS)))
    return max(1, int(round(delta * 1000.0)))


def build_layout(first_frames: list[np.ndarray]) -> Layout:
    target_height = min(frame.shape[0] for frame in first_frames)
    camera_widths = tuple(
        max(1, int(round(frame.shape[1] * target_height / frame.shape[0])))
        for frame in first_frames
    )
    top_width = sum(camera_widths)
    panel_height = 260
    return Layout(
        target_height=target_height,
        camera_widths=camera_widths,
        top_width=top_width,
        panel_height=panel_height,
    )


def resize_frame(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    if frame.shape[1] == width and frame.shape[0] == height:
        return frame
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)


def draw_camera_label(frame: np.ndarray, label: str) -> None:
    cv2.rectangle(frame, (10, 10), (200, 40), (20, 20, 20), thickness=-1)
    cv2.putText(
        frame,
        label,
        (18, 33),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (240, 240, 240),
        2,
        cv2.LINE_AA,
    )


def grip_status(value: float) -> str:
    return "OPEN" if value >= GRIPPER_OPEN_THRESHOLD else "CLOSE"


def format_joint_line(prefix: str, joint_values: np.ndarray, start: int, end: int) -> str:
    labels = [f"J{idx + 1}:{joint_values[idx]:7.2f}" for idx in range(start, end)]
    return f"{prefix} " + "  ".join(labels)


def render_info_panel(
    width: int,
    height: int,
    qpos: np.ndarray,
    frame_idx: int,
    num_frames: int,
    timestamp: float,
    episode_name: str,
    playback_fps: float,
) -> np.ndarray:
    panel = np.full((height, width, 3), 24, dtype=np.uint8)
    left_deg = np.degrees(qpos[:6])
    left_grip = float(qpos[6])
    right_deg = np.degrees(qpos[7:13])
    right_grip = float(qpos[13])

    lines = [
        f"{episode_name} | frame {frame_idx + 1}/{num_frames} | t={timestamp:.3f}s | playback={playback_fps:.2f} FPS",
        "Controls: space play/pause | a prev | d next | j -10 | l +10 | q/esc quit",
        format_joint_line("Left ", left_deg, 0, 3),
        format_joint_line("Left ", left_deg, 3, 6),
        f"Left  gripper: {grip_status(left_grip):<5} ({left_grip:.2f})",
        format_joint_line("Right", right_deg, 0, 3),
        format_joint_line("Right", right_deg, 3, 6),
        f"Right gripper: {grip_status(right_grip):<5} ({right_grip:.2f})",
    ]

    y = 32
    for idx, line in enumerate(lines):
        scale = 0.7 if idx == 0 else 0.62
        color = (255, 255, 255) if idx != 1 else (180, 180, 180)
        cv2.putText(panel, line, (18, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)
        y += 30

    return panel


def render_frame(frame_idx: int, episode: EpisodeData, layout: Layout, playback_fps: float) -> np.ndarray:
    frames = []
    for cam_name, target_width in zip(CAMERA_ORDER, layout.camera_widths):
        image = decode_image(episode.images[cam_name][frame_idx], cam_name, frame_idx)
        image = resize_frame(image, target_width, layout.target_height)
        draw_camera_label(image, CAMERA_TITLES[cam_name])
        frames.append(image)

    mosaic = np.concatenate(frames, axis=1)
    info_panel = render_info_panel(
        width=layout.top_width,
        height=layout.panel_height,
        qpos=episode.qpos[frame_idx],
        frame_idx=frame_idx,
        num_frames=episode.num_frames,
        timestamp=float(episode.timestamps[frame_idx]),
        episode_name=episode.path.stem,
        playback_fps=playback_fps,
    )
    return np.vstack([mosaic, info_panel])


def export_video(
    episode: EpisodeData,
    layout: Layout,
    output_path: Path,
    playback_fps: float,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame0 = render_frame(0, episode, layout, playback_fps)
    height, width = frame0.shape[:2]
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        playback_fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer: {output_path}")

    try:
        for idx in range(episode.num_frames):
            frame = frame0 if idx == 0 else render_frame(idx, episode, layout, playback_fps)
            writer.write(frame)
            if idx % 100 == 0 or idx == episode.num_frames - 1:
                print(f"Exporting video: {idx + 1}/{episode.num_frames}")
    finally:
        writer.release()


def interactive_view(
    episode: EpisodeData,
    layout: Layout,
    start_frame: int,
    fps_override: float | None,
    playback_fps: float,
) -> None:
    if start_frame >= episode.num_frames:
        raise ValueError(f"--start-frame {start_frame} is out of range for {episode.num_frames} frames")

    current = start_frame
    playing = True
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    try:
        while True:
            canvas = render_frame(current, episode, layout, playback_fps)
            cv2.imshow(WINDOW_NAME, canvas)
            delay = compute_frame_delay_ms(current, episode.timestamps, fps_override) if playing else 0
            key = cv2.waitKey(delay) & 0xFF

            if key in (ord("q"), 27):
                break
            if key == ord(" "):
                playing = not playing
                continue
            if key == ord("a"):
                current = max(0, current - 1)
                playing = False
                continue
            if key == ord("d"):
                current = min(episode.num_frames - 1, current + 1)
                playing = False
                continue
            if key == ord("j"):
                current = max(0, current - 10)
                playing = False
                continue
            if key == ord("l"):
                current = min(episode.num_frames - 1, current + 10)
                playing = False
                continue

            if playing:
                if current < episode.num_frames - 1:
                    current += 1
                else:
                    playing = False
    finally:
        cv2.destroyAllWindows()


def default_output_path(episode_path: Path) -> Path:
    return episode_path.with_name(f"{episode_path.stem}_visualized.mp4")


def main() -> int:
    args = parse_args()
    ensure_runtime_dependencies()
    episode_path = resolve_episode_path(args.data_dir, args.episode, args.file)
    episode = load_episode(episode_path)
    playback_fps = infer_fps(episode.timestamps, args.fps)

    first_frames = [
        decode_image(episode.images[cam_name][0], cam_name, 0)
        for cam_name in CAMERA_ORDER
    ]
    layout = build_layout(first_frames)

    print(f"Loaded episode: {episode.path}")
    print(f"Frames: {episode.num_frames}")
    print(f"Playback FPS: {playback_fps:.2f}")

    if args.save_video:
        output_path = args.out.expanduser().resolve() if args.out is not None else default_output_path(episode.path)
        export_video(episode, layout, output_path, playback_fps)
        print(f"Saved video to: {output_path}")

    interactive_view(
        episode=episode,
        layout=layout,
        start_frame=args.start_frame,
        fps_override=args.fps,
        playback_fps=playback_fps,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)
