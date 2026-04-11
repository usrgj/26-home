from common.skills.agv_api import agv

agv.start()
print(agv.get_map_data("4-11"))
agv.stop()