#!/usr/bin/env python3 
import rospy
from std_msgs.msg import UInt16MultiArray
import subprocess
import json
import time
import matplotlib.pyplot as plt
import threading
import os

timingFlag = False
pieceFlag = False


class Im2YoloManager:

    def __init__(self) -> None:
        rospy.init_node("visualizer_ideal")
        self.sub = rospy.Subscriber ("/state_inf_ideal", UInt16MultiArray, callback=self.pose_callback)
        rospy.loginfo("Node has been started.")
        self.proc = subprocess.Popen(['python3', './visualization_script.py'], 
                            stdin=subprocess.PIPE,
                            bufsize=1,
                            universal_newlines=True)
        
        self.id = -1
        if timingFlag:
            if os.path.exists('GTtiming.txt'):
                os.remove('GTtiming.txt')




    def pose_callback(self, state_info: UInt16MultiArray):
        milsec_st =  1000*time.perf_counter()
        self.id += 1
        if pieceFlag:
            with open('pieces.txt', 'a') as file: 
                    #file.write(f"{self.id}" + '\n')
                    data_string = ' '.join(str(e) for e in list(state_info.data[1:]))
                    file.write(data_string + '\n')

        json_data = self.convertGameState(list(state_info.data))

        json_data = json.loads(json_data)

        self.proc.stdin.write(json.dumps(json_data) + '\n')
        self.proc.stdin.flush()
        
        if timingFlag:
            finalStr = state_info.layout.dim[0].label + '/' + str(int(1000*time.perf_counter() - milsec_st)) 
            str_values = finalStr.split('/')
            if len(str_values) == 4:
                with open('GTtiming.txt', 'a') as file: 
                    file.write(' '.join(str_values) + '\n')

    def convertGameState(self,msgList):
        stateBoard = msgList[1::]

        pieces = ['bp', 'wR', 'wB', 'bQ', 'bK', 'bR', 'wp', 'bN', 'bB', 'wQ', 'wK', '--', 'wN', '--', '--']
        handStateList = ['P1s Hand on the board', 'P1s Move', 'P2s Hand on the board', 'P2s Move']
        state = []
        for i in range(8):
            row = []
            for j in range(8):
                row.append(pieces[stateBoard[i*8+j]])
            state.append(row[::-1])

        #self.id += 1
        data = {
        'id': self.id,
        'handState' : handStateList[msgList[0]],
        'state_board': state
        }

        return json.dumps(data)


if __name__ == '__main__':
    
    im2yoloman = Im2YoloManager()
    rospy.spin()

