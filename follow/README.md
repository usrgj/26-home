# 人物跟随系统 — 方案四：视觉 + LiDAR 融合 + 分层控制

## 项目结构

```
person_follower/
├── main.py                ← 完整版主入口 (Phase 4 最终形态)
├── config.py              ← 全局配置参数
├── robot_api.py           ← ★ 硬件抽象层 (你需要重点适配)
├── robot_api_example.py   ← ★ 基于 AGVManager 的适配参考
├── lidar_processor.py     ← LiDAR处理 (合并/差分/聚类/腿检测)
├── vision_detector.py     ← 视觉检测与ReID
├── sensor_fusion.py       ← EKF传感器融合
├── motion_controller.py   ← 运动控制 (PID + VFH)
├── state_machine.py       ← 分层控制状态机
├── map_preprocessor.py    ← 地图预处理工具 (log → .npy)
├── phase1_lidar_follow.py ← ★ Phase 1 最简版 (纯LiDAR跟随)
├── simulator.py           ← 离线仿真 (不需要真机)
├── maps/                  ← 预处理后的地图文件
│   ├── map_points.npy     ← 障碍物点云 (36,354点)
│   ├── map_meta.json      ← 地图元数据
│   ├── map_lines.json     ← 306条特征线
│   └── map_marks.json     ← 标记点
└── README.md
```

## 系统架构

```
[深度相机x4]─→ 人物检测(YOLO) ─→ ReID匹配 ─→ 视觉观测 ─┐
                                                          ├─→ EKF融合 ─→ 目标状态
[LiDAR x2] ─→ 合并360° ─→ 地图差分 ─→ 聚类 ─→ 腿检测 ──┘     │
                  │                                            ↓
                  └─→ 障碍物扇区 ──────────────────→ VFH避障  状态机
                                                      │      │
                                                      ↓      ↓
                                              运动控制器 ←── 模式选择
                                                  │
                                          ┌───────┼───────┐
                                          ↓       ↓       ↓
                                       直接控制  导航API  搜索旋转
```

## 快速开始

### 1. 安装依赖
```bash
pip install numpy scipy opencv-python
# 可选 (推荐): YOLO检测模型
pip install ultralytics
# 可选: 仿真可视化
pip install matplotlib
```

### 2. 预处理地图 (一次性)
```bash
python map_preprocessor.py <你的log文件> ./maps/
```

### 3. 运行离线仿真 (不需要真机，验证算法)
```bash
python simulator.py
```

### 4. Phase 1: 纯LiDAR跟随 (最快上真机)
只需在 `robot_api.py` 中实现 **3 个方法**:
- `get_robot_pose()`
- `get_lidar_scans()`
- `send_velocity()`

参考 `robot_api_example.py` 中基于 AGVManager 的示例。
```bash
python phase1_lidar_follow.py
```

### 5. 逐步升级到完整版
Phase 2 → 4 → 最终运行:
```bash
python main.py
```

## 各模块详细说明

### robot_api.py (★ 必须适配)
所有与机器人硬件的交互都封装在这个文件中。每个方法都有详细的注释和
示例代码，说明了预期的输入输出格式。你需要将占位实现替换为你自己
机器人SDK/API的实际调用。

### lidar_processor.py
- 合并前后两个LiDAR扫描为统一的360°点云
- 与静态地图做差分 (KD-Tree最近邻)，提取动态点
- DBSCAN聚类，筛选腿部候选
- 两两配对腿部 → 生成人物候选

### vision_detector.py
- 调用YOLO检测人物边界框
- 从深度图获取距离 (中心区域中位数)
- 像素坐标+深度 → 3D世界坐标
- 颜色直方图ReID (可替换为深度学习ReID)

### sensor_fusion.py
- 匀速运动模型的EKF
- 状态: [x, y, vx, vy]
- 支持视觉/LiDAR双传感器更新
- 数据关联 (最近邻 + 门控距离)
- 无观测时匀速预测 (coasting)

### motion_controller.py
- PID控制线速度 (距离误差) 和角速度 (方向误差)
- VFH避障: 360°扇区化 → 障碍密度 → 安全方向选择
- 多级速度调节: 紧急停止 / 减速 / 全速

### state_machine.py
三种模式:
1. **DIRECT_FOLLOW**: 目标可见，直接PID+避障跟随
2. **NAV_FOLLOW**: 目标丢失，切导航API走全局路径
3. **SEARCH**: 到达目标位置仍未找到，原地旋转搜索

## 分阶段实现路径 (推荐)

### Phase 1: 纯LiDAR跟随 ← 从这里开始!
**文件**: `phase1_lidar_follow.py`
**需实现**: `get_robot_pose()`, `get_lidar_scans()`, `send_velocity()`
**功能**: 跟随最近的动态LiDAR聚类，比例控制，基础障碍物停止
**验证**: 让一个人在机器人前方走动，机器人应该跟上
**预计工作量**: 1-2天 (主要在适配AGVManager接口)

### Phase 2: 加入视觉检测 + ReID
**需新增**: `get_camera_frame()`, 安装 YOLO
**功能**: 用视觉锁定特定目标人物，不再跟随"最近的"
**验证**: 多人场景中只跟随指定的人

### Phase 3: EKF融合
**功能**: 融合视觉和LiDAR观测，目标短暂遮挡时靠预测维持跟踪
**验证**: 目标被柱子挡住1-2秒后仍能恢复跟随

### Phase 4: 分层状态机 + 导航
**需新增**: `navigate_to()`, `cancel_navigation()`, `get_navigation_status()`
**功能**: 目标丢失时切换到导航模式，支持过门、拐弯
**验证**: 目标穿过门，机器人能跟着过去

### 离线仿真
**文件**: `simulator.py`
**任何阶段**都可以用仿真来调参和验证算法逻辑，不需要连接真机。
