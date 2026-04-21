# 人物跟随子系统

当前目录实现的是 `task1` 中“接包后跟随主人前往放包区域”的人物跟随子系统。  
它不是一个纯演示模块，真实接入点在 `task1/states/follow_and_place.py`，该状态会创建 `FollowRunner` 并循环调用 `start()` / `step()` / `stop()`。

本 README 以当前仓库代码为准，重点回答三件事：

- 真实运行链路是什么
- README 旧方案描述和当前实现有哪些偏差
- 现场调参时应该先看哪些参数，哪些参数实际上并没有生效

## 当前目录结构

```text
follow/
├── __init__.py
├── main.py
├── runner.py
├── config.py
├── robot_api.py
├── lidar_processor.py
├── vision_detector.py
├── sensor_fusion.py
├── motion_controller.py
├── state_machine.py
├── map_preprocessor.py
├── pull_map.py
├── motion_controller（复件）.py
├── state_machine（复件）.py
└── README.md
```

说明：

- 实际运行导入的是 `runner.py`、`state_machine.py`、`motion_controller.py` 这些正式文件。
- `motion_controller（复件）.py` 和 `state_machine（复件）.py` 不是当前运行链路的一部分，不要把它们当成主实现。

## 从任务入口看真实调用关系

比赛主流程里，`task1/states/follow_and_place.py` 会这样使用本子系统：

1. 创建 `FollowRunner`
2. 调用 `runner.start()` 尝试锁定主人
3. 在循环中持续调用 `runner.step()`
4. 外层状态决定何时停止跟随
5. 跟随后执行机械臂放包动作

独立调试入口是：

```bash
python task1/behaviors/follow/main.py
```

`main.py` 本身很薄，只负责日志、Ctrl+C 和调用 `FollowRunner`。

## 总体架构

当前实现的真实数据流如下：

```text
AGV push / cameras / lidars
        ↓
     RobotAPI
        ↓
  ┌───────────────┬────────────────┐
  ↓               ↓                ↓
LiDAR人物候选    视觉目标检测+ReID   障碍物扇区
  ↓               ↓                ↓
  └──────────────→ EKF 融合 ←──────┘
                    ↓
               TargetState
                    ↓
            MotionController
                    ↓
              StateMachine
                    ↓
     直接速度控制 / 导航API / 原地搜索
```

其中真正负责主循环调度的是 `runner.py`。

## 真实运行流程

### 1. 初始化阶段

`FollowRunner.start()` 会依次做这些事：

1. 初始化 `RobotAPI`、LiDAR 处理器、视觉检测器、EKF、运动控制器、状态机
2. 加载静态地图
3. 重置视觉目标模板和 EKF 状态
4. 等待首帧有效机器人位姿，避免用默认原点锁人
5. 尝试通过视觉从当前画面锁定目标
6. 状态机切到 `DIRECT_FOLLOW`

默认锁定策略不是“指定某个人”，而是：

- 从指定相机当前帧中检测所有人
- 选择最近的人
- 将其外观特征保存为目标模板

如果启动时没锁定成功，运行过程中会在第一次看到人时自动把最近的人锁成目标。

### 2. 每个 step 周期的执行顺序

`FollowRunner.step()` 的真实执行顺序如下：

1. 拉取一帧 AGV 推送
2. 从推送中读取机器人位姿
3. 读取双 LiDAR 数据
4. LiDAR 生成人物候选，同时生成避障扇区
5. 按固定周期读取视觉检测结果
6. 如果视觉中识别到目标，先用视觉更新 EKF
7. 用 LiDAR 候选与当前目标做关联，关联成功则再更新 EKF
8. 如果视觉和 LiDAR 都没有观测，则仅做 EKF 预测
9. 取出 `TargetState`
10. 用 `MotionController` 计算跟随速度
11. 用 `StateMachine` 决定当前是直接跟随、导航跟随、搜索还是丢失
12. 根据状态下发速度或导航命令

## 各模块细节

### RobotAPI

`robot_api.py` 负责把底层硬件能力整理成跟随子系统可直接调用的接口。

当前已接入的能力包括：

- AGV 推送位姿
- 双 LiDAR 扫描
- 头部和胸部相机
- 速度控制
- 自由导航 API

关键点：

- `get_state()` 每周期拉一帧新的 AGV 推送，失败时抛异常
- `get_robot_pose()` 对空位姿做了兜底，优先返回上一次有效位姿
- `get_lidar_scans()` 直接使用 AGV 返回的 `install_info`
- `get_camera_frame()` 会在相机未启动时自动启动
- `navigate_to()`、`cancel_navigation()`、`get_navigation_status()` 已接入

这意味着本项目已经不是 README 旧版本里说的“等待你自己实现硬件接口”的阶段，而是已经针对当前机器人环境做了实装。

### LiDAR 处理链

`lidar_processor.py` 的完整处理流程是：

1. 把两个 LiDAR 的 beams 合并到统一机器人坐标系
2. 把点变到世界坐标系
3. 和静态地图做 KD-Tree 最近邻差分
4. 把差分后的动态点做 DBSCAN 聚类
5. 依据聚类半径筛选腿部候选
6. 将两条腿配对成人物候选
7. 同时把全部 LiDAR 点转成避障扇区

细节说明：

- 合并时真正使用的是每帧 LiDAR 数据里携带的 `install_x` 和 `install_yaw`，不是硬编码常量。
- 地图差分是本模块把“墙、桌腿、固定家具”从实时雷达里剔除的关键步骤。
- DBSCAN 输出的每个聚类会计算中心和半径。
- 如果聚类半径落在腿的经验范围内，会被标记为 `is_leg_candidate`。
- 两条腿之间的距离如果在允许范围内，就配成一个人，人物位置取两腿中点。
- 没配对上的单腿也会作为低置信度人物候选保留。

### 视觉检测与 ReID

`vision_detector.py` 的真实流程是：

1. 从 `PRIMARY_CAMERAS` 指定的相机列表中逐个取帧
2. 用 YOLO 检出所有 `person`
3. 对每个 bbox 在深度图中心区域取中位数深度
4. 用内参把像素中心投影到相机坐标系
5. 再根据相机安装位姿转成机器人坐标和世界坐标
6. 从人物裁剪图提取颜色直方图特征
7. 用特征和已锁定目标做相似度匹配
8. 给最佳匹配打上 `is_target=True`

当前 ReID 不是深度模型，而是基于 HSV 颜色直方图：

- 上半身直方图
- 下半身直方图
- 拼成 128 维特征
- 最后做归一化并用点积当余弦相似度

优点是实现简单，坏处是对衣服颜色、光照和相机视角比较敏感。

### 传感器融合

`sensor_fusion.py` 是一个 4 维匀速模型 EKF，状态为：

```text
[x, y, vx, vy]
```

流程如下：

1. 如果系统还没初始化，第一次有效观测直接初始化 EKF
2. 视觉观测到目标时，先按观测时间推进预测，再用视觉位置更新
3. LiDAR 候选不会全部更新，只会先做一次数据关联，选出最像目标的那个
4. 若视觉和 LiDAR 都没观测，则只做预测，进入 coasting
5. 输出 `TargetState`，其中包含当前位置、速度、朝向、预测位置和是否仍有效

LiDAR 关联策略：

- 优先拿最近 1 秒内的视觉锚点作为关联中心
- 如果没有新鲜视觉锚点，就用 EKF 当前估计位置
- 默认门控半径是 0.8 米
- 如果已经连续一段时间没观测，门控会缓慢放大

这部分很关键，因为多人的 LiDAR 候选并不会自己知道谁是主人，真正把 LiDAR 锁到“对的人”上，靠的是视觉锚点加 EKF 位置门控。

### 运动控制

`motion_controller.py` 同时负责：

- 跟随距离控制
- 朝向控制
- 基于 LiDAR 扇区的反应式避障
- 速度平滑和加速度限幅

控制过程：

1. 根据 `TargetState` 计算机器人到目标的距离和角度误差
2. 如果目标速度明显大于 0.1m/s，优先朝预测位置追，不直接追当前点
3. 用 VFH 从障碍物扇区里找到最接近期望方向的安全方向
4. 检查该方向上的障碍物净空
5. 根据目标距离误差算线速度 PID
6. 根据安全方向误差算角速度 PID
7. 如果角速度过大，再削弱线速度
8. 用最大加速度做限幅，避免指令跳变

特殊逻辑：

- 如果目标已经在机器人后方很远，机器人只转不前进
- 如果目标距离比设定跟随距离还近，允许轻微倒车
- 如果前方净空太小，会直接急停

### 分层状态机

`state_machine.py` 定义了 5 个状态：

- `IDLE`
- `DIRECT_FOLLOW`
- `NAV_FOLLOW`
- `SEARCH`
- `LOST`

其中真正用于主流程的是后 4 个。

切换逻辑：

- 启动后默认进入 `DIRECT_FOLLOW`
- 如果目标长时间不可见，从 `DIRECT_FOLLOW` 切到 `NAV_FOLLOW`
- 如果直接跟随中长期被避障压得走不动，也会切到 `NAV_FOLLOW`
- 如果导航到目标附近还没找回人，则切到 `SEARCH`
- `SEARCH` 超时后切到 `LOST`
- 只要重新稳定看到目标，就切回 `DIRECT_FOLLOW`

这里“目标可见”的定义不是单纯 `is_valid`，而是：

- `target.is_valid == True`
- 且 `target.is_coasting == False`

这意味着：

- EKF 预测期虽然目标仍可能被认为有效
- 但只要连续一小段时间没有新观测，就已经会被状态机当成“不可见”

## 当前实现与旧方案描述的偏差

下面这些点在调试前必须先知道。

### 1. 旧文档写的是 4 路深度相机，当前代码只用 2 路

当前 `config.py` 中的 `PRIMARY_CAMERAS` 只有：

- `head`
- `chest`

### 2. YOLO 在当前实现里不是可选依赖

`VisionDetector` 初始化时会直接：

```python
from ultralytics import YOLO
self._model = YOLO("yolov8s.pt")
```

没有 `ultralytics` 或模型文件不可用时，视觉模块会直接受影响。

### 3. 地图路径当前是硬编码绝对路径，而且默认路径在当前机器上并不存在

当前配置里：

```python
MAP_POINTS_NPY_PATH = "/home/blinx/26-home/task1/behaviors/follow/maps/map_points.npy"
```

如果这个路径不对，`LidarProcessor` 不会加载静态地图，后果是：

- LiDAR 地图差分失效
- 所有 LiDAR 点都会被当成动态点
- 动态点聚类更容易把家具、墙角、门框边缘等误当做人

上机前应先修正这个路径，或者确保运行 `map_preprocessor.py` 后把地图文件放到正确位置。

### 4. 有些参数虽然写在 config.py 里，但当前主链路并不真正使用

这些参数名看起来像可调，但当前代码并没有按它们驱动主流程：

- `DETECTION_INTERVAL_MS`
- `VFH_SECTOR_ANGLE`
- `LIDAR_FRONT_X`
- `LIDAR_FRONT_YAW`
- `LIDAR_REAR_X`
- `LIDAR_REAR_YAW`
- `LIDAR_FRONT_Z`
- `LIDAR_REAR_Z`
- `ROBOT_WHEEL_BASE`

原因分别是：

- 视觉频率真正由 `FollowRunner(..., vision_interval=3)` 控制，不是毫秒配置。
- VFH 当前固定用 72 个扇区，没有读取 `VFH_SECTOR_ANGLE` 来决定扇区数。
- LiDAR 安装参数实际来自 AGV 返回的 `install_info`。
- 轮距当前没有参与控制器计算。

### 5. 还有一部分关键阈值写死在代码里，不在 config.py

例如：

- LiDAR 关联默认门限 `0.8m`
- 视觉去重距离 `0.5m`
- 预测前视时间 `0.5s`
- 视觉深度过滤上限 `15m`

如果要把这些也变成可调参数，需要继续重构代码。

## 关键配置项怎么影响行为

本节按调参时最常用的类别来解释。

### 机器人物理与速度限制

位于 `config.py`：

- `ROBOT_RADIUS`
- `ROBOT_MAX_LINEAR_VEL`
- `ROBOT_MAX_ANGULAR_VEL`
- `MAX_LINEAR_ACCEL`
- `MAX_ANGULAR_ACCEL`

影响：

- `ROBOT_RADIUS` 越大，控制器认为机器人越胖，越容易提早减速或急停。
- 最大线速度和角速度决定控制输出的硬上限。
- 最大加速度决定速度响应的“肉”还是“冲”。

建议：

- 先确保 `ROBOT_RADIUS` 真实可靠，再调避障距离。
- 如果机器人跟随不够流畅但方向没问题，优先看加速度限幅。

### 视觉检测与目标锁定

常用参数：

- `DETECTION_CONFIDENCE_THRESHOLD`
- `REID_SIMILARITY_THRESHOLD`
- `PRIMARY_CAMERAS`
- `CAMERAS[...]` 中的相机位姿

影响：

- 检测阈值高，误检少，但漏检会变多。
- ReID 阈值高，不容易串人，但更容易在转身、遮挡、光照变化时丢人。
- 相机外参不准会让人物世界坐标偏移，进而影响 EKF 和控制。

建议：

- 多人场景容易锁错人时，先检查锁定流程是否让主人站在最前面。
- 如果“看得到人但一直锁不上”，优先检查 ReID 阈值和相机外参。
- 如果位置明显飘，先核对 `fx/fy/ppx/ppy` 与深度单位，再核对相机安装角。

### LiDAR 人物检测

常用参数：

- `CLUSTER_EPS`
- `CLUSTER_MIN_POINTS`
- `CLUSTER_MAX_POINTS`
- `LEG_MIN_RADIUS`
- `LEG_MAX_RADIUS`
- `LEG_PAIR_MIN_DIST`
- `LEG_PAIR_MAX_DIST`
- `MAP_DIFF_THRESHOLD`

影响：

- `CLUSTER_EPS` 太小会把一条腿分裂成碎点，太大又会把腿和附近障碍物粘起来。
- `CLUSTER_MIN_POINTS` 太大时，远处的人腿会被丢掉。
- `LEG_MIN/MAX_RADIUS` 决定什么样的聚类会被当作腿。
- `LEG_PAIR_MAX_DIST` 太大时，容易把两个不相关聚类配成一个人。
- `MAP_DIFF_THRESHOLD` 太小会保留太多静态点，太大会把真实动态点吸回静态地图。

建议：

- 先保证静态地图正确加载，再调聚类参数。
- 如果家具、门框总被当做人，先看地图差分，再看 `CLUSTER_EPS` 和腿半径范围。
- 如果远处主人总丢，先降低 `CLUSTER_MIN_POINTS` 或放宽腿配对距离。

### EKF 融合与短时遮挡恢复

常用参数：

- `EKF_PROCESS_NOISE_POS`
- `EKF_PROCESS_NOISE_VEL`
- `EKF_MEASUREMENT_NOISE_LIDAR`
- `EKF_MEASUREMENT_NOISE_VISION`
- `EKF_MAX_COAST_TIME`

影响：

- 过程噪声越大，预测更灵活，也更容易抖。
- 观测噪声越大，系统越不信对应传感器。
- `EKF_MAX_COAST_TIME` 越长，系统越愿意靠预测撑住短时丢失。

建议：

- 目标被遮挡后恢复困难，先看观测噪声和过程噪声是否平衡。
- 视觉明显更准时，应让视觉噪声小于 LiDAR 噪声。
- 如果预测飘得厉害但状态又不肯失效，应降低 `EKF_MAX_COAST_TIME`。

### 跟随距离与 PID

常用参数：

- `FOLLOW_DISTANCE`
- `FOLLOW_DISTANCE_TOLERANCE`
- `FOLLOW_ANGLE_TOLERANCE`
- `PID_LINEAR_KP`
- `PID_LINEAR_KI`
- `PID_LINEAR_KD`
- `PID_ANGULAR_KP`
- `PID_ANGULAR_KI`
- `PID_ANGULAR_KD`

影响：

- `FOLLOW_DISTANCE` 决定理想跟随间距。
- 距离容差越大，机器人越容易“停着不补”；越小则更频繁来回修正。
- 线速度 PID 主要影响远近跟随感。
- 角速度 PID 主要影响“盯人”是否利索。

建议：

- 跟随发冲，先减小线速度 `KP` 或增大距离容差。
- 左右摆头严重，先减小角速度 `KP` 或增加 `KD`。
- 不要一开始就动 `KI`，当前默认 `KI=0` 是比较稳妥的。

### 避障

常用参数：

- `OBSTACLE_DANGER_DIST`
- `OBSTACLE_SLOW_DIST`
- `OBSTACLE_AVOID_DIST`
- `ROBOT_RADIUS`

影响：

- `OBSTACLE_AVOID_DIST` 决定哪些扇区会被当成拥挤扇区参与 VFH 方向选择。
- `OBSTACLE_DANGER_DIST` 决定什么时候直接急停。
- `OBSTACLE_SLOW_DIST` 决定什么时候进入减速区。

建议：

- 机器人老是保守停住，先检查 `ROBOT_RADIUS` 是否偏大。
- 机器人会蹭到桌角或门框，再增大危险距离和减速距离。

### 状态切换

常用参数：

- `TARGET_LOST_TIMEOUT`
- `TARGET_RECOVER_TIMEOUT`
- `STUCK_TIMEOUT`
- `STUCK_SPEED_THRESHOLD`
- `STUCK_MIN_TARGET_DIST`
- `NAV_GOAL_REACHED_DIST`
- `SEARCH_ROTATION_SPEED`
- `SEARCH_TIMEOUT`

影响：

- `TARGET_LOST_TIMEOUT` 越短，越容易切导航；越长，越依赖直接跟随。
- `TARGET_RECOVER_TIMEOUT` 越短，导航中一看到人就越快切回直跟。
- `STUCK_TIMEOUT` 越短，越容易把“避障减速”判成卡住。
- `SEARCH_TIMEOUT` 越短，越快宣布目标彻底丢失。

建议：

- 目标短时被遮挡就立刻切导航，增大 `TARGET_LOST_TIMEOUT`。
- 导航过程中频繁抖动切换，适当增大 `TARGET_RECOVER_TIMEOUT`。
- 门口或狭窄区域总是误判卡住，适当增大 `STUCK_TIMEOUT` 或降低 `STUCK_SPEED_THRESHOLD` 的敏感度。

## 推荐调试顺序

如果你要现场把这个功能调稳，建议按下面顺序来，不要一上来同时改十几个参数。

### 第 1 步：先确认基础链路通

先确认这些前提：

- AGV 推送里有有效 `x/y/angle`
- 双 LiDAR 都有数据
- `head` 和 `chest` 相机能出彩色和深度
- YOLO 能正常加载
- 地图文件存在且路径正确

如果这些前提没满足，后面的调参意义不大。

### 第 2 步：先让视觉锁人稳定

目标是：

- 启动时总能锁到前方主人
- 多人场景下不容易串人

优先检查：

- `PRIMARY_CAMERAS`
- 相机外参
- `DETECTION_CONFIDENCE_THRESHOLD`
- `REID_SIMILARITY_THRESHOLD`

### 第 3 步：再让 LiDAR 人物候选稳定

目标是：

- 家具和门框不要频繁变成人
- 主人腿部能形成稳定候选

优先检查：

- 地图差分是否生效
- `MAP_DIFF_THRESHOLD`
- `CLUSTER_EPS`
- 腿半径和腿间距参数

### 第 4 步：再调 EKF 和状态切换

目标是：

- 人被短时遮挡不会立刻丢
- 重新出现后能快速接上

优先检查：

- `EKF_*`
- `TARGET_LOST_TIMEOUT`
- `TARGET_RECOVER_TIMEOUT`

### 第 5 步：最后调跟随体感和避障体感

目标是：

- 跟随距离自然
- 转身不发飘
- 不过度保守，也不蹭障碍物

优先检查：

- `FOLLOW_DISTANCE`
- 线速度和角速度 PID
- `ROBOT_RADIUS`
- `OBSTACLE_*`
- `MAX_*_ACCEL`

## 常见现象与优先排查方向

### 现象：总是跟错人

优先排查：

- 启动锁定时主人是不是离机器人最近
- `REID_SIMILARITY_THRESHOLD` 是否过低
- 视觉坐标是否漂得太远导致后续 LiDAR 关联串人

### 现象：家具总被识别成动态目标

优先排查：

- 静态地图是否真的加载成功
- `MAP_DIFF_THRESHOLD` 是否不合适
- 聚类和腿半径范围是否过宽

### 现象：明明还看到人，却频繁切到导航

优先排查：

- 视觉频率是否太低
- `TARGET_LOST_TIMEOUT` 是否太短
- EKF 是否过早进入 coasting

### 现象：导航和直跟来回抖动切换

优先排查：

- `TARGET_RECOVER_TIMEOUT` 是否太短
- 视觉检测是否间歇性抖动
- 目标检测坐标是否跳变过大

### 现象：机器人总是保守停住不敢走

优先排查：

- `ROBOT_RADIUS` 是否过大
- `OBSTACLE_DANGER_DIST` / `OBSTACLE_SLOW_DIST` 是否过保守
- 狭窄环境中 `STUCK_*` 是否过于敏感

### 现象：跟随时忽快忽慢、很“肉”

优先排查：

- `MAX_LINEAR_ACCEL`
- `PID_LINEAR_KP`
- `FOLLOW_DISTANCE_TOLERANCE`

## 当前版本的几个重要提醒

1. 当前视觉频率主要由 `FollowRunner(vision_interval=3)` 控制，不是 `DETECTION_INTERVAL_MS`。
2. 当前 LiDAR 安装位姿主要取自硬件返回数据，不是 `config.py` 里的前后雷达安装常量。
3. 当前 VFH 扇区数固定为 72，`VFH_SECTOR_ANGLE` 只是配置中存在，并未主导扇区划分。
4. 当前地图路径需要尽快改成当前机器真实可用的位置。
5. 当前 ReID 只是颜色直方图，现场服装、光照和遮挡变化大时要留出足够容错。

## 若后续要继续改进，优先级建议

如果后面继续增强这个子系统，建议优先做这些事：

1. 把当前写死在代码里的关键阈值也提到 `config.py`
2. 修正地图文件路径和地图加载流程，避免依赖单一绝对路径
3. 给视觉频率统一成一个真正生效的配置项
4. 给 LiDAR 关联增加显式置信度或速度一致性约束
5. 若多人干扰强，再考虑把当前简单颜色 ReID 换成更稳的模型

## 相关文件

- `task1/states/follow_and_place.py`
- `task1/behaviors/follow/main.py`
- `task1/behaviors/follow/runner.py`
- `task1/behaviors/follow/config.py`
- `task1/behaviors/follow/robot_api.py`
- `task1/behaviors/follow/lidar_processor.py`
- `task1/behaviors/follow/vision_detector.py`
- `task1/behaviors/follow/sensor_fusion.py`
- `task1/behaviors/follow/motion_controller.py`
- `task1/behaviors/follow/state_machine.py`
