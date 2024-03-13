#!/usr/bin/env python3 
import rospy
from std_msgs.msg import UInt16MultiArray
from std_msgs.msg import MultiArrayDimension
import time
import os

def stateManager(BoxLimits, pieceList, GridLimits, HandState):

    if not hasattr(stateManager, "state"):
        stateManager.state = 1
        stateManager.map = evaluateMap(BoxLimits, pieceList, GridLimits)

    if stateManager.state == 1:
        if HandState:
            stateManager.state = 0
    elif stateManager.state == 3:
        if HandState:
            stateManager.state = 2
    elif stateManager.state == 0:
        if not HandState:
            stateManager.map = evaluateMap(BoxLimits, pieceList, GridLimits)  
            stateManager.state = 3  
    elif stateManager.state == 2:
        if not HandState:
            stateManager.map = evaluateMap(BoxLimits, pieceList, GridLimits)
            stateManager.state = 1

    return stateManager.map, stateManager.state

def evaluateMap (BoxLimits, pieceList, GridLimits):
    pieceLocs = PieceLocator(BoxLimits)
    return BoardEstimator(pieceLocs, pieceList, GridLimits)

def PieceLocator(BoxLimits):

    pieceLocs = []

    for box in BoxLimits:
        cen_x = .5*box[0] + .5*box[2]
        cen_y = .5*box[1] + .5*box[3]
        cen = [cen_x, cen_y]
        pieceLocs.append(cen)
    
    return pieceLocs

def dst(pt1, pt2):
    return ((pt1[0]-pt2[0])**2 + (pt1[1]-pt2[1])**2 )**.5

def area(pt1, pt2, pt3):
    l1 = dst(pt1,pt2)
    l2 = dst(pt2,pt3)
    l3 = dst(pt1,pt3)
    s = (l1+l2+l3)/2
    return (s*(s-l1)*(s-l2)*(s-l3))**.5

def inside(pt, pt1, pt2, pt3, pt4):
     #print(area(pt, pt1, pt2) + area(pt, pt3, pt2) + area(pt, pt4, pt3) + area(pt, pt1, pt4))
     #print(area(pt3, pt1, pt2) + area(pt1, pt4, pt3))
     return not area(pt, pt1, pt2) + area(pt, pt3, pt2) + area(pt, pt4, pt3) + area(pt, pt1, pt4) - area(pt3, pt1, pt2) - area(pt1, pt4, pt3)>3


def BoardEstimator(pieceLocs, pieceList, GridLimits):
    board = [[14 for x in range(8)] for i in range(8)]
    for piece, pieceLoc in zip(pieceList, pieceLocs):
    
        for row in range(8):
            for col in range(8):
                ind0 = 9*row + col
                ind1 = ind0 + 1
                ind2 = ind0 + 9
                ind3 = ind0 + 10
                
                if inside(pieceLoc, GridLimits[ind0], GridLimits[ind1], GridLimits[ind3], GridLimits[ind2] ):
                    board[row][col] = piece
                
    return board


class Im2YoloManager:

    def __init__(self) -> None:
        rospy.init_node("yolo2state_ideal")
        self.pub =rospy.Publisher("/state_inf_ideal", UInt16MultiArray, queue_size=10)
        self.sub = rospy.Subscriber ("/yolo_res_ideal", UInt16MultiArray, callback=self.pose_callback)
        rospy.loginfo("Node has been started.")
        self.grids = []
        self.id = 1
        if os.path.exists('GTpieces.txt'):
            os.remove('GTpieces.txt')



    def pose_callback(self, yol: UInt16MultiArray):
        milsec_st =  1000*time.perf_counter()
        dimString = MultiArrayDimension()
        cmd = UInt16MultiArray()
        msg_in = list(yol.data)
        n_obj = (len(msg_in)-163)//5
        hand_flag = msg_in[0]
        obj_list = msg_in[1:(n_obj+1)]
        obj_boundries = []
        for i in range(n_obj):
            obj_boundries.append([ msg_in[4*i+n_obj+1], 
                                  msg_in[4*i+n_obj+2], 
                                  msg_in[4*i+n_obj+3], 
                                  msg_in[4*i+n_obj+4] ])
        
        if sum(msg_in[-162:]):
            grids = []
            for j in range(81):
                grids.append([msg_in[-2*j-2], msg_in[-2*j-1]])
            grids.reverse()
            self.grids = grids

        if len(self.grids):
            game_map,game_state = stateManager(obj_boundries, obj_list, self.grids, hand_flag)
            out_msg = []
        
            out_msg.append(game_state)

            out_msg += sum(game_map,[])
            #
            cmd.data = out_msg

            #print(f"Veysel {out_msg}")
            stream_string = ' '.join(map(str, out_msg[1:]))  # This creates a comma-separated string from the list
            with open('GTpieces.txt', 'a') as file:
                file.write(stream_string + '\n')

            dimString.label = yol.layout.dim[0].label + '/' + str(int(1000*time.perf_counter() - milsec_st)) 
            cmd.layout.dim = [dimString]


            self.pub.publish(cmd)

if __name__ == '__main__':
    
    im2yoloman = Im2YoloManager()
    rospy.spin()

