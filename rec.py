from socket import *
import re
import json
import time
import threading

class Blinx_AGVPush():
    def __init__(self):
        try:
            self.socket_agv_push = socket(AF_INET, SOCK_STREAM)
            self.socket_agv_push.setsockopt(SOL_SOCKET, SO_KEEPALIVE, 1)  # 在客户端开启心跳维护
            self.socket_agv_push.connect(("192.168.192.5", 19301))
            self.BUF_SIZE = 20000

            self.statu = True

            self.current_station = None  # AGV站点数据
            self.blocked = None  # 赋值AGV是否被阻挡
            self.battery_level = None  # 机器人电量
            self.charging = None  # 机器人是否在充电
            self.DI = None  # DI数据
            self.DO = None  # DO数据
            self.brake = None  # 是否抱闸
            self.is_stop = None  # 机器人底盘是否禁止
            self.fatals = None  # 严重报警
            self.errors = None  # 报警
            self.warnings = None  # 警告
            self.x = None  # x坐标
            self.y = None  # y坐标
            self.angle = None  # 角度坐标
            self.emergency = None  # 急停状态

            self.agv_statu = None   # AGV状态

            self.target_station = "LM3"   # 目标站点
	    
            self.agv_arrived = "No"   # AGV是否到达

            # 开线程，防止堵塞
            self.clientR = threading.Thread(target=self.state_msg)
            self.clientR.start()
        except Exception as e:
            print(e)
    # 通讯关闭方法
    def state_close(self):
        self.socket_agv_push.close()

    # 消息接收
    def state_msg(self):
        while True:
            try:
                # 消息接受
                data = self.socket_agv_push.recv(self.BUF_SIZE)
                data = str(re.findall(r"({.+})", str(data)))
                data = str(data).replace("[\'", "")
                data = str(data).replace("\']", "")
                if len(data) == 0 or data == None:
                    print('AGV服务断开，请检查AGV后重启程序')
                else:
                    product = json.loads(data)
                    print(data)
                    self.current_station = product['current_station']  # AGV站点数据
                    self.blocked = product['blocked']  # 赋值AGV是否被阻挡
                    self.battery_level = product['battery_level']  # 机器人电量
                    self.charging = product['charging']  # 机器人是否在充电
                    self.DI = product['DI']  # DI数据
                    self.DO = product['DO']  # DO数据
                    self.brake = product['brake']  # 是否抱闸
                    self.is_stop = product['is_stop']  # 机器人底盘是否禁止
                    self.fatals = product['fatals']  # 严重报警
                    self.errors = product['errors']  # 报警
                    self.warnings = product['warnings']  # 警告
                    self.x = product['x']  # x坐标
                    self.y = product['y']  # y坐标
                    self.angle = product['angle']  # 角度坐标
                    self.emergency = product['emergency']  # 急停状态

                    #print(self.current_station)
                    #print(self.is_stop)
                    # 判断是否存在警告
                    if len(self.fatals) > 0 or len(self.errors) > 0:
                        self.agv_statu = 'alarm'  # 将AGV状态改为报警状态
                    # 判断机器人是否是在运行状态
                    elif not self.is_stop:
                        self.agv_statu = 'running'  # 将AGV状态改为运行状态
                    else:
                        self.agv_statu = 'ready'  # 将AGV状态改为准备状态

                    """
                    判断AGV是否到达点位
                    条件一：判断客户端是否发送了站点指令   
                    条件二：判断客户端发送的站点是否与AGV站点相同
                    条件三：判断AGV底盘是否禁止
                    条件四：判断AGV状态不为告警
                    """
                    #print(self.target_station)
                    if len(self.target_station) > 0 and self.is_stop and \
                            self.current_station == self.target_station:
                        self.agv_arrived = 'yes'
                        self.target_station = ""
                        #print(self.agv_arrived)
                    elif self.current_station != self.target_station and len(self.target_station) > 0:
                        self.agv_arrived = 'no'
                        #print(self.agv_arrived)
            except Exception as e:
                print("AGV19301:", e)
                if self.statu:
                    data = "{\"interval\":500,\"included_fields\":[\"current_station\",\"x\",\"y\",\"angle\",\"blocked\",\"battery_level\"" \
                           ",\"charging\",\"emergency\",\"DI\",\"DO\",\"brake\",\"is_stop\",\"fatals\",\"errors\",\"warnings\"]}"
                    data = ''.join([hex(ord(c)).replace('0x', '') for c in data])
                    length = hex(int(len(data) / 2)).replace('0x', '')
                    num = (8 - len(length))
                    for i in range(num):
                        length = '0' + length
                    msg = "5A010001" + length + "2454000000000000" + data
                    msg = msg.upper()
                    self.socket_agv_push.send(bytes.fromhex(msg))
                self.statu = False

    # 向AGV发送消息
    def state_send(self, msg):
        self.socket_agv_push.send(msg.encode('utf-8'))

#agv_rec = Blinx_AGVPush()
#agv_rec.state_msg()