import subprocess
import signal
import os
import time

play_bag_path = os.path.expanduser('~') + '/exper_data/7091_25hz.bag'
output_bag_path = os.path.expanduser('~') + '/exper_data/7091_add_0_5_noise.bag'

# 启动roslaunch
roslaunch_cmd = ["roslaunch", "relative_ctrl", "ikfom_uav.launch"]
roslaunch_proc = subprocess.Popen(roslaunch_cmd)

# 启动rosbag play
rosbag_play_cmd = ["rosbag", "play", play_bag_path]
rosbag_play_proc = subprocess.Popen(rosbag_play_cmd)

# 启动rosbag record
rosbag_record_cmd = ["rosbag", "record", "-a", "-O", output_bag_path]
rosbag_record_proc = subprocess.Popen(rosbag_record_cmd)

# 等待rosbag play完成
rosbag_play_proc.wait()

# 停止rosbag record
rosbag_record_proc.send_signal(signal.SIGINT)

# 等待rosbag record停止
time.sleep(5)

# 停止roslaunch
roslaunch_proc.send_signal(signal.SIGINT)
roslaunch_proc.wait()

print("所有进程已停止。")
