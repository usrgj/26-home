汇总了一些常用的api

### 平动

1. port: 19206

2. cmd id: 0BEF

3. 描述:平动，以固定速度直线运动固定距离。根据使用，y方向的值无效，该指令只能给x方向赋值。

4. | 字段名 | 类型   | 描述                                                         | 可缺省 |
   | :----- | :----- | :----------------------------------------------------------- | :----- |
   | dist   | number | 直线运动距离, 绝对值, 单位: m                                | 否     |
   | vx     | number | 机器人坐标系下 X 方向运动的速度, 正为向前, 负为向后, 单位: m/s | 是     |
   | ~~vy~~ | number | 机器人坐标系下 Y 方向运动的速度, 正为向左, 负为向右, 单位: m/s | 是     |
   | mode   | number | 0 = 里程模式(根据里程进行运动), 1 = 定位模式, 若缺省则默认为里程模式 | 是     |

### 转动

1. port: 19206

2. cmd id: 0BF0

3. 描述:转动, 以固定角速度旋转固定角度

4. | 字段名 | 类型   | 描述                                                         | 可缺省 |
   | :----- | :----- | :----------------------------------------------------------- | :----- |
   | angle  | number | 转动的角度(机器人坐标系), 绝对值, 单位 rad, 可以大于 2π      | 否     |
   | vw     | number | 转动的角速度(机器人坐标系), 正为逆时针转, 负为顺时针转 单位 rad/s | 否     |
   | mode   | number | 0 = 里程模式(根据里程进行运动), 1 = 定位模式, 若缺省则默认为里程模式 | 是     |

### 查询机器人激光点云数据

1. port: 19204
2. cmd id: 03F1
3. 描述:查询机器人激光点云数据。下面是响应数据
4. | 字段名 | 类型  | 描述                         |
   | :----- | :---- | :--------------------------- |
   | lasers | array | 激光点云数据, 数据示例见下文 |

### 查询机器人位置

1. port: 19204
2. cmd id: 03EC
3. 描述:查询机器人位置，下面为响应
4. | 字段名          | 类型   | 描述                                                         | 可缺省 |
   | :-------------- | :----- | :----------------------------------------------------------- | :----- |
   | x               | number | 机器人的 x 坐标, 单位 m                                      | 是     |
   | y               | number | 机器人的 y 坐标, 单位 m                                      | 是     |
   | angle           | number | 机器人的 angle 坐标, 单位 rad                                | 是     |
   | confidence      | number | 机器人的定位置信度, 范围 [0, 1]                              | 是     |
   | current_station | string | 离机器人最近站点的 id（该判断比较严格，机器人必须很靠近某一个站点，这个距离可以通过参数配置中的 CurrentPointDist修改，默认为 0.3m。如果机器人与所有站点的距离大于这个距离，则这个字段为空字符。如果想要获取上一个完成任务的站点，可以取 finished_path 数组中的最后一个元素）![img](https://seer-group.feishu.cn/space/api/box/stream/download/asynccode/?code=NmVhOTBjM2NjMGE4ZDRiYmRhMDZkZmNlMmNmYmFkMDlfM2Yzd2RSa0tOemlzbFpTNklPM1Q4V2Y2bGFUcm5tWFFfVG9rZW46Qm9RbGJKT1dnb2d0TmF4U2JPQ2M4c2pvblRtXzE3NzM3MjgyMzc6MTc3MzczMTgzN19WNA)注意：上图这个例子中，机器人从AP1到AP2，中间靠近AP3时，current_station会返回3。 | 是     |
   | last_station    | string | 机器人上一个所在站点的 id                                    | 是     |
   | loc_method      | number | 0 = 自然轮廓定位 1 = 反光柱定位 2 = 二维码定位 3 = 里程计模式 4 = 3D 定位 5 = 天码定位 6 = 特征定位 7 = 3D 特征定位 8 = 3D KF定位 | 是     |
   | ret_code        | number | API 错误码                                                   | 是     |
   | create_on       | string | API 上传时间戳                                               | 是     |
   | err_msg         | string | 错误信息                                                     | 是     |

### 机器人推送

1. port: 19301
2. cmd id: 4B65
3. 描述:该 `API` 用于机器人主动推送数据到所连接的客户端。

### 配置机器人推送端口

1. port: 19301
2. cmd id: 9300
3. 描述:个人认为，当需要高频获取某些信息时，可以在这里设置好，避免使用高频请求。使用后记得改回默认值（我们这个底盘是500ms）include和exclude不能同时设置
4. | **字段名**      | **类型**      | **描述**               | **可缺省** |
   | :-------------- | :------------ | :--------------------- | :--------- |
   | interval        | Interger      | 消息推送时间间隔（ms） | 是         |
   | included_fields | array[string] | 设置消息中包含的字段   | 是         |
   | excluded_fields | array[string] | 设置消息中排除的字段   | 是         |

### 1

1. port: 
2. cmd id: 
3. 描述:

