'''
底盘通讯的使用demo
API参考
https://seer-group.feishu.cn/wiki/X0LSw4NRRiXTtGksoMxcQ4knnxh
'''
from __future__ import annotations
from agv_api import AGVManager, agv_manager
import time
# from camera.config import CAMERA_CHEST
# from camera import camera_manager

PORT_STATUS = 19204
PORT_CONTROL = 19205
PORT_NAVIGATION = 19206
PORT_CONFIGUE = 19207
PORT_OTHER = 19210
PROT_PUSH = 19301

# ── Demo ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    with AGVManager() as mgr:

        # 查询（等响应）
        # mgr.send(19210, "17D4",data={"slam_type":1, "real_time": True})
        # result = mgr.query(port=19204, cmd_id="0401")
        # info = result.response["data"]
        # print(info.get('slam_status'))
        
        # 获取激光雷达数据
        result = mgr.query(19204, "03F1")
        info = result.response["data"]
        print(info.get('lasers'))
        time.sleep(4)
        
        # mgr.send(19210, "17D5")

        #自由导航测试
        # result = mgr.query(19206, "0BEB", data={
        #                                 "freeGo": {
        #                                     "theta": 1.0, # 弧度
        #                                     "x": 2.052,
        #                                     "y": 4.740
        #                                 },
        #                                 "id": "SELF_POSITION"
        #                                 })
        # print(result.response["data"])

        
        # 获取区域id
        # result = mgr.query(19204, "03F3")
        # print(result.response["data"])

        # result = mgr.query(19207, "0FAB",data={"map_name":"3_15"})
        # print(result.response["data"])

        # 查询机器人位置
        # result = mgr.query(19204, "03EC")
        # info = result.response["data"]

        # 开环运动控制
        # result = mgr.query(PORT_CONTROL, "07DA", data={"vx": -0.2, "vy": 0.0, "w": 0.0, "duration": 0})
        # print(result.response)
        # time.sleep(2)
        # vx = 0.0
        # w = 0.0

        # # 测试高频接受运动指令
        # for i in range(20):

        #     result = mgr.query(PORT_CONTROL, "07DA", data={"vx": vx, "vy": 0.0, "w": w, "duration": 0})
        #     print(result.response)
        #     vx+=1/10
        #     w+=1/10
        #     time.sleep(0.5)

        # 机器人主动推送 修改推送配置需要等待几秒
        
        
        # time.sleep(2)

        # print(mgr.poll().response["data"])
        # for i in range(5):
        #     print(mgr.poll().response["data"])
        #     time.sleep(0.3)
        
        # result = mgr.query(19301,"2454", data={"interval": 50,
        #                               "included_fields": [
        #                                 "x", "y", "angle",
        #                                  ]})
        # print(result)

        # time.sleep(3)
        # for i in range(10):
        #     print(mgr.poll().response["data"])
        #     time.sleep(0.1)

        #获取指定地图的json数据
        # result = mgr.query(PORT_CONFIGUE, "0FAB", data={"map_name":"3_15"})
        # print(result.response["data"])

        # None

    
    # mgr = agv_manager
    # mgr.start()
    # try:
    #     None
    # finally:
    #     mgr.stop()

        
    print("[主线程] 程序正常结束")
