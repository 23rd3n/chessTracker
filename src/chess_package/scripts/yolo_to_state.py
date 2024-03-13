#!/usr/bin/env python3 
import rospy
from std_msgs.msg import UInt16MultiArray
from std_msgs.msg import MultiArrayDimension
import time

def dst(pt1, pt2):
    return ((pt1[0]-pt2[0])**2 + (pt1[1]-pt2[1])**2 )**.5

def area(pt1, pt2, pt3):
    l1 = dst(pt1,pt2)
    l2 = dst(pt2,pt3)
    l3 = dst(pt1,pt3)
    s = (l1+l2+l3)/2
    return (abs(s*(s-l1)*(s-l2)*(s-l3)))**.5


def inside(pt, pt1, pt2, pt3, pt4):
     return not area(pt, pt1, pt2) + area(pt, pt3, pt2) + area(pt, pt4, pt3) + area(pt, pt1, pt4) - area(pt3, pt1, pt2) - area(pt1, pt4, pt3)>3


def PieceLocator(BoxLimits):

    pieceLocs = []

    for box in BoxLimits:
        cen_x = .5*box[0] + .5*box[2]
        cen_y = .5*box[1] + .5*box[3]
        cen = [cen_x, cen_y]
        pieceLocs.append(cen)
    
    return pieceLocs


def evaluateMap (BoxLimits, pieceList, GridLimits, probs):

    board = [[14 for _ in range(8)] for _ in range(8)]
    prob_board = [[[j//14 for j in range(15)] for _ in range(8)] for _ in range(8)]

    for thisPieces, boxlimit, probOfPieces, thisGridLimits  in zip(pieceList, BoxLimits, probs, GridLimits  ):
        
        pieceLocs = PieceLocator(boxlimit)
        
        for piece, pieceLoc, pieceProb in zip(thisPieces, pieceLocs, probOfPieces):

            for row in range(8):
                for col in range(8):
                    ind0 = 9*row + col
                    ind1 = ind0 + 1
                    ind2 = ind0 + 9
                    ind3 = ind0 + 10
                    
                    if inside(pieceLoc, thisGridLimits[ind0], thisGridLimits[ind1], thisGridLimits[ind3], thisGridLimits[ind2] ):
                        #print(pieceLoc, thisGridLimits[ind0], thisGridLimits[ind1], thisGridLimits[ind3], thisGridLimits[ind2] )
                        prob_board[row][col][piece] +=  pieceProb
    
    board = [[ prob_board[a][b].index(max(prob_board[a][b])) for b in range(8)] for a in range(8)]
                
    return board


def stateManager(BoxLimits, pieceList, GridLimits, HandState, probs):

    if not hasattr(stateManager, "state"):
        stateManager.state = 1
        stateManager.map = evaluateMap(BoxLimits, pieceList, GridLimits, probs)

    if stateManager.state == 1:
        if HandState:
            stateManager.state = 0
    elif stateManager.state == 3:
        if HandState:
            stateManager.state = 2
    elif stateManager.state == 0:
        if not HandState:
            stateManager.map = evaluateMap(BoxLimits, pieceList, GridLimits, probs)  
            stateManager.state = 3  
    elif stateManager.state == 2:
        if not HandState:
            stateManager.map = evaluateMap(BoxLimits, pieceList, GridLimits, probs)
            stateManager.state = 1

    return stateManager.map, stateManager.state


class Im2YoloManager:

    def __init__(self) -> None:
        rospy.init_node("yolo2state")
        self.pub =rospy.Publisher("/state_inf", UInt16MultiArray, queue_size=10)
        self.sub = rospy.Subscriber ("/yolo_res", UInt16MultiArray, callback=self.pose_callback)
        rospy.loginfo("Node has been started.")
        self.grids = []
        self.id = 1


    def pose_callback(self, yol: UInt16MultiArray):
        milsec_st =  1000*time.perf_counter()
        dimString = MultiArrayDimension()
        cmd = UInt16MultiArray() 
        msg_in = list(yol.data) 
        
        HandState = msg_in[0]
        ch_board_flag = msg_in[1]
        n1 = msg_in[2]
        n2 = msg_in[3]
        
        obj_list1 = msg_in[4:(n1+4)]
        obj_list2 = msg_in[(n1+4):(n1+n2+4)]
        pieceList = [ obj_list1,obj_list2 ]
        
        obj_boundries1 = [msg_in[i:(i+4)] for i in range(n1 + n2 + 4, 5*n1 + n2 + 4, 4)]
        obj_boundries2 = [msg_in[i:(i+4)] for i in range(5*n1 + n2 + 4, 5*n1 + 5*n2 + 4, 4)]
        BoxLimits = [ obj_boundries1,obj_boundries2 ]
        
        probs1 = msg_in[(5*n1 + 5*n2 + 4):(6*n1 + 5*n2 + 4) ]
        probs2 = msg_in[(6*n1 + 5*n2 + 4):(6*n1 + 6*n2 + 4) ]
        probs = [probs1, probs2]
        
        if ch_board_flag:
            cb1 = msg_in[-324:-162]
            cb1 = [cb1[i:(i+2)] for i in range(0, 162, 2)]
            cb2 = msg_in[-162:]
            cb2 = [cb2[i:(i+2)] for i in range(0, 162, 2)]
            self.grids = [cb1, cb2]
        
        game_map,game_state = stateManager(BoxLimits, pieceList, self.grids , HandState, probs)
        out_msg = []
        out_msg.append(game_state)
        out_msg += sum(game_map,[])

        cmd.data = out_msg
        print("serden ben")
        #print(f"Veysel {out_msg}")
        # self.id += 1
        # name = "yolo2stat" + str(self.id) + ".txt"
        # stream_string = ', '.join(map(str, out_msg))  # This creates a comma-separated string from the list
        # with open(name, 'w') as file:
        #     file.write(stream_string)
        dimString.label = yol.layout.dim[0].label + '/' + str(int(1000*time.perf_counter() - milsec_st)) 
        cmd.layout.dim = [dimString]

        self.pub.publish(cmd)

if __name__ == '__main__':
    
    im2yoloman = Im2YoloManager()
    rospy.spin()

