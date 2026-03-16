'''
底盘通讯的使用demo
API参考
https://seer-group.feishu.cn/wiki/X0LSw4NRRiXTtGksoMxcQ4knnxh
'''
from __future__ import annotations
from agv_api import AGVManager
import time

# ── Demo ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    with AGVManager() as mgr:

        # 查询（等响应）
        # mgr.send(19210, "17D4",data={"slam_type":1, "real_time": True})
        # result = mgr.query(port=19204, cmd_id="0401")
        # info = result.response["data"]
        # print(info.get('slam_status'))
        
        # result = mgr.query(19204, "03F1")
        # info = result.response["data"]
        # print(info.get('lasers'))
        # time.sleep(4)
        
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

        result = mgr.query(19207, "0FAB",data={"map_name":"3_15"})
        print(result.response["data"])

        
    print("[主线程] 程序正常结束")
