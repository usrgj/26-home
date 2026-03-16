'''
底盘通讯的使用demo
API参考
https://seer-group.feishu.cn/wiki/X0LSw4NRRiXTtGksoMxcQ4knnxh
'''
from __future__ import annotations
import time
import uuid
from agv_manager import AGVManager

# ── Demo ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    with AGVManager() as mgr:

        print("\n" + "=" * 50)
        print("主线程开始执行，客户端线程在后台运行")
        print("=" * 50 + "\n")

        # 查询（等响应）
        print("[主线程] 查询机器人信息...")
        result = mgr.query(port=19204, cmd_id="03E8")
        if result and result.ok:
            info = result.response["data"]
            print(f"[主线程] 型号: {info.get('model')}，地图: {info.get('current_map')}")
        elif result:
            print(f"[主线程] 查询失败: {result.error}")

        # 发送导航（不等响应）
        task_id = uuid.uuid4().hex[:8].upper() #这行用于生成 简短、唯一、伪随机标识符
        print(f"\n[主线程] 发送导航指令 task_id={task_id}")



        # 主循环：做自己的事，定期 poll AGV 推送
        print("\n[主线程] 进入主循环，每秒 poll 一次 AGV 推送")
        for i in range(5):
            print(f"[主线程] 第 {i + 1} 次业务逻辑")
            time.sleep(1)
            while msg := mgr.poll():
                data_str = msg.response["data"] if msg.ok else msg.error
                print(f"  [poll] 端口{msg.port} cmd={msg.cmd_id}: {data_str}")

    print("[主线程] 程序正常结束")

'''
使用with是为了线程安全
可以自行实例化，但记得start和安全stop
mgr = AGVManager()
mgr.start()
try:
    # 任务逻辑
finally:
    mgr.stop()


如果在多个文件中使用agv，比如未来可以采用每个人完成一个任务，而不是按功能分组的任务分配方法
那么可以在agv_manager中添加单例，在每个文件中导入这同一个单例即可。

'''