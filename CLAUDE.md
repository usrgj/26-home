# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

RoboCup@Home 2026 service robot — person-following system with vision+LiDAR fusion. The robot uses a Seer (仙知) AGV chassis communicating over a custom TCP binary protocol, Intel RealSense depth cameras, and a Kinco servo motor for a slide mechanism.

Language: Python 3.10+. All comments and logs are in Chinese (中文).

## Running

```bash
# Full system (starts AGV, cameras, then follow loop)
python main.py

# Phase 1: LiDAR-only follow (minimal hardware needed)
python -c "from follow.phase1_lidar_follow import main; main()"

# AGV communication test
python test.py

# Slide motor control test
python slide_control/demo.py
```

## Dependencies

```bash
pip install numpy scipy opencv-python
pip install ultralytics    # YOLO detection (optional)
pip install pyrealsense2   # Intel RealSense cameras
pip install pyserial        # Slide motor Modbus RTU
```

## Architecture

### Startup flow (`main.py`)
1. `agv_manager.start()` — connects to all AGV TCP ports in parallel
2. `camera_manager.start(serial)` — starts RealSense cameras by serial number
3. `follow.main()` — runs the 10Hz follow loop

### AGV Communication (`agv_api/`)
- **`agv_protocol.py`** — Frame codec for Seer AGV binary protocol (header `5A010001` + length + cmd_id + reserved + JSON body). Response cmd_id = request cmd_id + 10000 (decimal).
- **`agv_client.py`** — Persistent TCP client per port with background recv thread. `send()` (fire-and-forget) and `request()` (blocking query). Callback registration via `on(cmd_id, fn)` for push messages.
- **`agv_manager.py`** — Thread-safe facade over multiple `AGVClient` instances. Main thread uses `query()` / `send()` / `poll()` via command queues. Module-level singleton: `agv_manager`.

Key AGV ports: 19204 (status), 19205 (control), 19206 (navigation), 19207 (config), 19301 (push). Push messages use cmd_id `4B65`; subscribe with cmd_id `2454` on port 19301.

### Follow System (`follow/`)
10Hz main loop with 6 pipeline stages:
1. Robot pose (from AGV push data)
2. LiDAR processing → dynamic person candidates (map diff + DBSCAN clustering + leg detection)
3. Vision detection (YOLO + depth + ReID, runs every 3rd frame)
4. EKF sensor fusion (state: `[x, y, vx, vy]`)
5. State machine: IDLE → DIRECT_FOLLOW ↔ NAV_FOLLOW → SEARCH → LOST
6. Motion control (PID + VFH obstacle avoidance) → `send_velocity`

- **`robot_api.py`** — Hardware abstraction layer bridging `follow/` modules to `agv_api` and `camera`. Data classes: `RobotPose`, `LidarScan`, `CameraFrame`, etc. Velocity control via AGV port 19205 cmd `07DA`.
- **`config.py`** — All tunable parameters (robot dimensions, LiDAR install offsets, PID gains, EKF noise, state machine timeouts). This is the primary file for parameter tuning.

### Camera (`camera/`)
- **`camera_manager.py`** — Module-level `CameraManager` singleton managing `RealSenseCamera` instances by serial number. Provides aligned color+depth frames.
- **`config.py`** — Camera serial numbers (HEAD, CHEST, LEFT, RIGHT).

### Slide Control (`slide_control/`)
Kinco servo motor control via Modbus RTU over serial (`/dev/ttyACM0`, 38400 baud). CiA402 state machine for enable/disable. Supports position, velocity, and homing modes.

## Key Conventions

- AGV API commands use hex string cmd_ids (e.g., `"03E8"`, `"07DA"`)
- AGV default IP: `192.168.192.5`
- Robot pose `theta`/`angle` is in radians; LiDAR angles are in degrees
- Map preprocessing: `python follow/map_preprocessor.py <log_file> ./maps/` generates `map_points.npy` for LiDAR background subtraction
- Camera serials are hardcoded in `camera/config.py`
- The follow system gets real-time pose from AGV push subscription (50ms interval) rather than polling
