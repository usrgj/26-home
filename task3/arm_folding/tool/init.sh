# 相机，导轨，定位，机械臂初始化
conda run -n final2 python /home/blinx/桌面/angle.py - 14000
conda run -n final2 python /home/blinx/mhl_test/26/slide_control/slide_locate.py
conda run -n final2 python /home/blinx/mhl_test/26/slide_control/slide_locate.py
conda run -n @home python /home/blinx/mhl_test/26/locate.py
conda run -n final2 python arm_init.py