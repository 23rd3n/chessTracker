#!/usr/bin/env python3
import subprocess
import os
import signal
import sys
import time

def signal_handler(sig, frame):
    print("SIGINT received, terminating processes...")
    for p in processes:
        try:
            # Attempt graceful shutdown
            os.killpg(os.getpgid(p.pid), signal.SIGTERM)
        except OSError as e:
            print(f"Error terminating process {p.pid}: {e}")
    # Optional: wait a bit for processes to terminate gracefully
    time.sleep(2)
    # Ensure termination if still running
    for p in processes:
        try:
            os.killpg(os.getpgid(p.pid), signal.SIGKILL)
        except OSError as e:
            print(f"Error forcefully killing process {p.pid}: {e}")

def wait_for_output(process, text):
    while True:
        output = process.stdout.readline()
        if text in output:
            break


# Register the signal handler
signal.signal(signal.SIGINT, signal_handler)

processes = []
catkin = subprocess.Popen("catkin_make", shell=True)
catkin.wait()  # Wait for the command to complete
roscore = subprocess.Popen("roscore", shell=True)
processes.append(roscore)


print("Playing vis..")
vis_process = subprocess.Popen(
    "rosrun chess_package state_to_disp.py",
    shell=True,
    stdout=subprocess.PIPE, text=True
)
time.sleep(2)
#wait_for_output(vis_process, "started.")
processes.append(vis_process)


print("Playing tolo2stat..")
yolo2state_process = subprocess.Popen(
    "rosrun chess_package yolo_to_state.py",
    shell=True, stdout=subprocess.PIPE, text=True
)
time.sleep(2)
#wait_for_output(yolo2state_process, "started.")
processes.append(yolo2state_process)


print("Playing im2yolo..")
im2yolo_process = subprocess.Popen(
    "rosrun chess_package image_to_yolo.py",
    shell=True, stdout=subprocess.PIPE, text=True
)
time.sleep(2)
#wait_for_output(im2yolo_process, "started.")

processes.append(im2yolo_process)


print("Playing ros2im..")
ros2im_process = subprocess.Popen(
    "rosrun chess_package rosbag_to_image.py",
    shell=True, stdout=subprocess.PIPE, text=True
)
time.sleep(2)
#wait_for_output(ros2im_process, "started.")
processes.append(ros2im_process)

print("Playing Rosbag file...")
processes.append(subprocess.Popen(
            "rosbag play /home/student/Documents/GroupA/full_game_2_2024-01-15-18-13-53.bag", 
            shell=True, 
            stdout=subprocess.DEVNULL,  # Ignore standard output
            stderr=subprocess.DEVNULL   # Ignore standard error
        ))
# Keep the script running until a signal is received
signal.pause()