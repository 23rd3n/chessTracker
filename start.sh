#!/bin/bash

# Function to kill all subprocesses
cleanup() {
    echo "Caught SIGINT signal. Killing all subprocesses..."
    pkill -f -9 "rosbag_to_image.py"
    # Then, kill all other processes in the current process group
    kill 0
}

# Check for --no-accuracy argument
SKIP_IDEAL=0
for arg in "$@"; do
    if [[ $arg == "--no-accuracy" ]]; then
        SKIP_IDEAL=1
    fi
done

# Trap SIGINT (Ctrl+C) and call cleanup function
trap cleanup SIGINT

# Start roscore in the background
roscore &
ROS_PID=$! # Save PID of roscore
sleep 2 # Give roscore time to start

# Ensure we're in the right directory
cd pphauRos1_ws || exit

catkin_make &
sleep 1

if [ $SKIP_IDEAL -eq 0 ]; then
    echo "ideal viz"
    rosrun chess_package state_to_disp_ideal.py &
    sleep 1
fi

echo "viz"
rosrun chess_package state_to_disp.py &
sleep 1

if [ $SKIP_IDEAL -eq 0 ]; then
    echo "ideal yolo2state"
    rosrun chess_package yolo_to_state_ideal.py &
    sleep 1
fi

echo "yolo2state"
rosrun chess_package yolo_to_state.py &
sleep 1

if [ $SKIP_IDEAL -eq 0 ]; then
    echo "ideal im2yolo"
    rosrun chess_package image_to_yolo_ideal.py &
    sleep 1
fi

echo "im2yolo"
rosrun chess_package image_to_yolo.py &
sleep 10

echo "playing rosbag"

rosrun chess_package rosbag_to_image.py &
sleep 3

# Go back to the original directory
cd ..

# Play the rosbag file, waiting for it to finish
rosbag play FullGameEditedTopicName.bag

sleep 15
# After everything is done, or if SIGINT is caught, kill roscore and any other subprocesses
cleanup

# Wait for roscore to properly shut down
wait $ROS_PID

