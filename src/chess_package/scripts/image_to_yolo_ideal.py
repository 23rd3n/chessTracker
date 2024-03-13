#!/usr/bin/env python3 
import rospy
from std_msgs.msg import UInt16MultiArray
from std_msgs.msg import MultiArrayDimension
from ultralytics import YOLO
import numpy as np
import cv2 as cv
import torch.cuda
import time

######## myUtils.py ########
def RemoveArbitraryCorn(corners):

    sorted_corn = corners[np.argsort(corners[:, 1])]
    norms_diff = np.linalg.norm(sorted_corn[1:] - sorted_corn[:-1], axis=1)
    del_indx = np.where(norms_diff < 2)[0]
    #print("deleted_indxes",del_indx)
    sorted_corn = np.delete(sorted_corn,del_indx,axis=0)
    #print("detected corners after deletion",sorted_corn.shape)

    return sorted_corn

def findUnitVecs(sorted_corn,method="Median"):
    if method == "Median":
        diff_vec = np.diff(sorted_corn,axis=0)
        unit_x,unit_y = np.median(diff_vec[:,0]),np.median(diff_vec[:,1])
        unit_vec_hor = np.array([unit_x,unit_y])

        sorted_corn = sorted_corn[np.argsort(sorted_corn[:, 0])]
        diff_vec = np.diff(sorted_corn,axis=0)
        unit_x,unit_y = np.median(diff_vec[:,0]),np.median(diff_vec[:,1])
        unit_vec_vert = np.array([unit_x,unit_y])
    
    elif method == "Mean":
        diff_vec = np.diff(sorted_corn,axis=0)
        Q1 = np.percentile(diff_vec[:,0], 25)
        Q3 = np.percentile(diff_vec[:,0], 75)
        IQR = Q3 - Q1
        # Define the data points that are not outliers
        non_outlier_data = diff_vec[:,0][(diff_vec[:,0] >= Q1 - 1.5 * IQR) & (diff_vec[:,0] <= Q3 + 1.5 * IQR)]
        # Compute the average of the non-outlier data
        unit_x = np.mean(non_outlier_data)
        Q1 = np.percentile(diff_vec[:,1], 25)
        Q3 = np.percentile(diff_vec[:,1], 75)
        IQR = Q3 - Q1
        # Define the data points that are not outliers
        non_outlier_data = diff_vec[:,1][(diff_vec[:,1] >= Q1 - 1.5 * IQR) & (diff_vec[:,1] <= Q3 + 1.5 * IQR)]
        # Compute the average of the non-outlier data
        unit_y = np.mean(non_outlier_data)
        unit_vec_vert = np.array([unit_x,unit_y])

        sorted_corn = sorted_corn[np.argsort(sorted_corn[:, 1])]
        diff_vec = np.diff(sorted_corn,axis=0)
        Q1 = np.percentile(diff_vec[:,0], 25)
        Q3 = np.percentile(diff_vec[:,0], 75)
        IQR = Q3 - Q1
        # Define the data points that are not outliers
        non_outlier_data = diff_vec[:,0][(diff_vec[:,0] >= Q1 - 1.5 * IQR) & (diff_vec[:,0] <= Q3 + 1.5 * IQR)]
        # Compute the average of the non-outlier data
        unit_x = np.mean(non_outlier_data)
        Q1 = np.percentile(diff_vec[:,1], 25)
        Q3 = np.percentile(diff_vec[:,1], 75)
        IQR = Q3 - Q1
        # Define the data points that are not outliers
        non_outlier_data = diff_vec[:,1][(diff_vec[:,1] >= Q1 - 1.5 * IQR) & (diff_vec[:,1] <= Q3 + 1.5 * IQR)]
        # Compute the average of the non-outlier data
        unit_y = np.mean(non_outlier_data)
        unit_vec_hor = np.array([unit_x,unit_y])

    return unit_vec_hor, unit_vec_vert

def findMiddlePoint(sorted_corn,w,h):
    mid_pred = np.linalg.norm(sorted_corn-np.array([w/2,h/2]),axis = 1)
    mid_indx = np.argmin(mid_pred)
    middle_point = sorted_corn[mid_indx]
    
    return middle_point

def CompletePattern(sorted_corn,unit_vec_hor,unit_vec_vert,middle_point):
    for i in range(-4,5):
        for j in range(-4,5):
            pred_coor = i*unit_vec_vert + j*unit_vec_hor + middle_point
            norm = np.linalg.norm(pred_coor - sorted_corn, axis=1)

            if np.min(norm) >=25:
                indx = xy2lin(i,j)
                if indx>=sorted_corn.shape[0]:
                    return np.zeros((81,2),dtype=np.uint8)
                sorted_corn = np.insert(sorted_corn, indx, pred_coor, axis=0)
                #print(f"inserted {pred_coor}, to {indx}th index.")
    return sorted_corn
                
def DrawCheckerboard(img,corners):
    if not np.all(corners==0):
        for i in range(corners.shape[0]):
            cv.circle(img,(corners[i,0],corners[i,1]), 3, (0,0,255), -1)
            cv.putText(img, f'{i}', (corners[i,0],corners[i,1]), cv.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 1)


def xy2lin(i,j):
    return 9*(i+4)+(j+4)

def perfect_sort(corners):
    mat_corn = corners.reshape(9,9,2)
    for i in range(9):
        mat_corn[i] = mat_corn[i,np.argsort(mat_corn[i,:,0])[::-1]]

    return mat_corn.reshape(81,2)

######## myUtils.py END ########

######## Checkerboard_detect_funcs.py ########

block_size = 15
sober_cons = 3

def detectCheckerboardCoords(frame,bbox,method = "Median"):
    if frame.shape == 0:
        return np.zeros((81,2),dtype=np.uint8)

    h,w,c = frame.shape

    dst = cv.Canny(frame, 50, 200, None, 3)

    linesP = cv.HoughLinesP(dst, rho=1, theta=np.pi / 180, threshold=75, minLineLength=150, maxLineGap =50)
    empty_im = np.zeros_like(dst,dtype=np.uint8)

    if linesP is not None:
        for i in range(0, len(linesP)):
            l = linesP[i][0]
            cv.line(empty_im, (l[0], l[1]), (l[2], l[3]), (255,255,255), 1, cv.LINE_AA)


    dst = cv.cornerHarris(empty_im,block_size,sober_cons,0.03)
    dst = cv.dilate(dst,None)
    ret, dst = cv.threshold(dst,0.1*dst.max(),255,0)
    dst = np.uint8(dst)

    _, _, _, centroids = cv.connectedComponentsWithStats(dst)

    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.001)
    corners = cv.cornerSubPix(empty_im,np.float32(centroids),(15,15),(-1,-1),criteria)
    corners = np.int0(corners)

    sorted_corn = RemoveArbitraryCorn(corners=corners)
    unit_vec_hor, unit_vec_vert = findUnitVecs(sorted_corn=sorted_corn, method=method)
    middle_point = findMiddlePoint(sorted_corn,w,h)
    sorted_corn = CompletePattern(sorted_corn,unit_vec_hor,unit_vec_vert,middle_point)

    if np.all(sorted_corn == 0) or sorted_corn.shape[0]!=81:
        #print('Checkerboard pattern is not trusted,returning empty array.')
        return np.zeros((81,2), dtype=np.uint8)
    else:
        sorted_corn = perfect_sort(sorted_corn)
        return np.int0(sorted_corn+np.array([bbox[0],bbox[1]]))

    
######## Checkerboard_detect_funcs.py END ########
    
######## get_iou.py ########
    
def get_iou(a, b, epsilon=1e-5):

    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])

    width = (x2 - x1)
    height = (y2 - y1)

    if (width < 0) or (height < 0):
        return 0.0

    area_overlap = width * height

    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    area_combined = area_a + area_b - area_overlap

    iou = area_overlap / (area_combined + epsilon)
    return iou

######## get_iou.py END ########

######## get_iohand.py END ########

def get_iohand(a, b, epsilon=1e-5):

    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])

    width = (x2 - x1)
    height = (y2 - y1)

    if (width < 0) or (height < 0):
        return 0.0

    area_overlap = width * height

    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    #area_combined = area_a + area_b - area_overlap

    iohand = area_overlap / (area_b + epsilon)
    #print(f"ratio {iohand}")
    return iohand

######## get_iohand.py END ########
######## detectHands.py ########

def detectHands(bboxList, classList):
    boardId = 11
    handId = 13
    threshold = 0.15

    if boardId not in classList:
        assert "no board is found in frame!"
    if handId not in classList:
        return 0  # no hand in frame

    # there may be multiple hands in the frame
    handIndices = [idHand for idHand, className in enumerate(classList) if className == handId]
    handBoundingBoxes = [bboxList[index] for index in handIndices]

    # but there is only one checkerboard, return first occurence
    boardIndex = classList.index(boardId)
    boardBbox = bboxList[boardIndex]

    isHandDetected = [get_iohand(boardBbox, handBbox) >= threshold for handBbox in handBoundingBoxes]
    #print('handBoundingBoxes',handBoundingBoxes)
    #print('boardBbox',boardBbox)

    if any(isHandDetected):
        return 1

    return 0

######## detectHands.py END ########

######## YOLOfuncs.py ########

def YOLOdetect(model, frame):
    '''
    Tracking functions. Takes model and frame as input and returns the bbounding box 
    coordinations and class numbers.

    Inputs:
    model (YOLO module): Tracking/detecting model.
    frame (numpy array): Current frame.

    Outputs:
    bboxes (numpy array): Bounding box coordinates. 
    classes (numpy array): Class numbers.
    '''
    
    results = model.predict(frame)
    bboxes = results[0].boxes.cpu().numpy().xyxy
    classes = results[0].boxes.cpu().numpy().cls.astype(np.int32)

    return bboxes, classes

def CheckerboardFrame(frame, bboxes, classes):
    indx = np.where(classes == 11)[0]
    ch_coords = np.int0(bboxes[indx][0])
    
    if ch_coords.shape == 0:
        ch_frame = np.array([])
    else:
        ch_frame = frame[ch_coords[1]:ch_coords[3],ch_coords[0]:ch_coords[2]]

    return ch_frame, ch_coords

def PieceDetection(bboxes, classes):
    mask = (classes != 13) & (classes != 11)
    indx = np.where(mask)[0]

    return np.int0(bboxes[indx]), classes[indx]

def concatStream(handflag, piece_classes, piece_bboxes, sorted_coord):
    stream = []
    stream.append(handflag)
    stream.extend(piece_classes.tolist())
    stream.extend(piece_bboxes.reshape(-1).tolist())
    if sorted_coord.shape != 0:
        stream.extend(sorted_coord.reshape(-1).tolist())

    return stream


######## YOLOfuncs.py END ########



class Im2YoloManager:

    def __init__(self) -> None:
        rospy.init_node("im2yolo_ideal")
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f'Using device: {device}')
        self.model = YOLO('/home/student/pphauRos1_ws/models/bests_acc.pt').to(device)

        self.pub =rospy.Publisher("/yolo_res_ideal", UInt16MultiArray, queue_size=10)
        self.sub = rospy.Subscriber ("/im", UInt16MultiArray, callback=self.pose_callback)
        rospy.loginfo("Node has been started.")
        self.id = 0


    def pose_callback(self, im: UInt16MultiArray):
        cmd = UInt16MultiArray() 
        cmdData = list(im.data) 
        milsec_st =  1000*time.perf_counter()
        dimString = MultiArrayDimension()
        cmd = UInt16MultiArray()
        frame_id = list(im.data) [-1]
        print(f"Frame id {frame_id}")
        #cmd.data = im.data #+ [61]
        img = cv.cvtColor(np.array(list(im.data)[:-1],dtype=np.uint8).reshape(720,1280,3),cv.COLOR_BGR2RGB)
        bboxes, classes = YOLOdetect(self.model, img)
        #print('classes',classes)
        handflag = detectHands(bboxes.tolist(), classes.tolist())
        if not handflag:
            ch_frame, bbox_ch = CheckerboardFrame(img, bboxes, classes)
            sorted_coord = detectCheckerboardCoords(ch_frame,bbox_ch)
            #DrawCheckerboard(img,sorted_coord)
            piece_bboxes, piece_classes = PieceDetection(bboxes, classes)
            stream = concatStream(handflag, piece_classes, piece_bboxes, sorted_coord)
            cmd.data = stream
            #print("ifdeyim")
        else:
            
            data_array = np.zeros((163,), dtype=np.uint8)
            data_array[0] = handflag

            # Assign to cmd.data
            cmd.data = data_array.tolist()
            #print(f"Elsedeyim {cmd.data}")
        dimString.label = im.layout.dim[0].label + '/' + str(int(1000*time.perf_counter() - milsec_st)) 
        cmd.layout.dim = [dimString]

        self.pub.publish(cmd)

        # TO DO: publish stream(list) #
        #print(f"Su {stream}")
        # self.id += 1
        # name = "im2yolo" + str(self.id) + ".txt"
        # stream_string = ', '.join(map(str, stream))  # This creates a comma-separated string from the list
        # with open(name, 'w') as file:
        #     file.write(stream_string)

        

if __name__ == '__main__':
    
    im2yoloman = Im2YoloManager()
    rospy.spin()