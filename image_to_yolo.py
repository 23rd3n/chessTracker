#!/usr/bin/env python3
from ultralytics import YOLO
import numpy as np
import cv2 as cv
import torch.cuda
import time

ch_flag = True

def __main__():
    #read the image, frame will be in the place of img
    img =  cv.imread("frame001845.png")
    #load the model (once)
    model = YOLO('best.pt')
    #run the model on frame
    bboxes1, classes1, probs1, bboxes2, classes2, probs2 = YOLOdetect(model, img1,img2)

    #get handflag
    handflag1 = detectHands(bboxes1.tolist(), classes1.tolist())
    handflag2 = detectHands(bboxes2.tolist(), classes2.tolist())


    piece_bboxes1, piece_classes1 = PieceDetection(bboxes1, classes1)
    piece_bboxes2, piece_classes2 = PieceDetection(bboxes2, classes2)

    if ch_flag:
        #get checkerboard coordinates
        ch_frame1, bbox_ch1 = CheckerboardFrame(img1, bboxes1, classes1)
        ch_frame2, bbox_ch2 = CheckerboardFrame(img2, bboxes2, classes2)

        sorted_coord1 = detectCheckerboardCoords(ch_frame1,bbox_ch1)
        sorted_coord2 = detectCheckerboardCoords(ch_frame2,bbox_ch2)

        #uncomment for visualization
        DrawCheckerboard(img1,sorted_coord1)
        DrawCheckerboard(img2,sorted_coord2)
        #get piece coordinates

        stream = concatStreamwcoord(handflag1,handflag2, piece_classes1, piece_classes2, piece_bboxes1,piece_bboxes2, probs1, probs2, sorted_coord1, sorted_coord2)
    else:
        concatStreamwcoord_nocoord(handflag1,handflag2, piece_classes1, piece_classes2, piece_bboxes1,piece_bboxes2, probs1, probs2)   
    
    print(stream)
    cv.imshow("img1",img1)
    cv.imshow("img2", img2)

    cv.waitKey()

######## myUtils.py ########
def RemoveArbitraryCorn(corners):

    sorted_corn = corners[np.argsort(corners[:, 1])]
    norms_diff = np.linalg.norm(sorted_corn[1:] - sorted_corn[:-1], axis=1)
    del_indx = np.where(norms_diff < 2)[0]
    print("deleted_indxes",del_indx)
    sorted_corn = np.delete(sorted_corn,del_indx,axis=0)
    print("detected corners after deletion",sorted_corn.shape)

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

def delExcessPieces(sorted_corn,unit_vec_hor,unit_vec_vert,middle_point):
    pred_mat = np.zeros((81,2),np.uint16)
    for i in range(-4,5):
        for j in range(-4,5):
            pred_mat[xy2lin(i,j),:] = np.clip(i*unit_vec_vert + j*unit_vec_hor + middle_point,0,100000)

    cross_norms_matrix = np.linalg.norm(pred_mat[:, np.newaxis, :] - sorted_corn, axis=2)
    probs = np.min(cross_norms_matrix,0) > 20
    indices = np.where(probs == True)
    sorted_corn = np.delete(sorted_corn,indices,axis=0)

    return sorted_corn

def CompletePattern(sorted_corn,unit_vec_hor,unit_vec_vert,middle_point):
    for i in range(-4,5):
        for j in range(-4,5):
            pred_coor = i*unit_vec_vert + j*unit_vec_hor + middle_point
            norm = np.linalg.norm(pred_coor - sorted_corn, axis=1)

            if np.min(norm) >=32:
                indx = xy2lin(i,j)
                if indx>=sorted_corn.shape[0]:
                    print(f"try inserted {pred_coor}, to {indx}th index.")
                    return np.zeros((81,2),dtype=np.uint8)
                sorted_corn = np.insert(sorted_corn, indx, pred_coor, axis=0)
                print(f"inserted {pred_coor}, to {indx}th index.")
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

    linesP = cv.HoughLinesP(dst, rho=1, theta=np.pi / 180, threshold=75, minLineLength=150, maxLineGap =75)
    empty_im = np.zeros_like(dst,dtype=np.uint8)

    if linesP is not None:
        for i in range(0, len(linesP)):
            l = linesP[i][0]
            cv.line(empty_im, (l[0], l[1]), (l[2], l[3]), (255,255,255), 1, cv.LINE_AA)
            cv.line(frame, (l[0], l[1]), (l[2], l[3]), (255,255,255), 1, cv.LINE_AA)
            


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
    sorted_corn = delExcessPieces(sorted_corn, unit_vec_hor,unit_vec_vert,middle_point)
    sorted_corn = CompletePattern(sorted_corn,unit_vec_hor,unit_vec_vert,middle_point)

    if np.all(sorted_corn == 0) or sorted_corn.shape[0]!=81:
        print('Checkerboard pattern is not trusted,returning empty array.')
        return np.zeros((81,2), dtype=np.uint8)
    else:
        print('Checkerboard pattern detected.')
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

def YOLOdetect(model, frame1, frame2):
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
    
    results = model.predict([ frame1, frame2])
    boxes1 = results[0].boxes.cpu().numpy()
    bboxes1 = boxes1.xyxy
    classes1 = boxes1.cls.astype(np.int32)
    probs1 = (boxes1.conf*100).astype(np.uint8)

    boxes2 = results[1].boxes.cpu().numpy()
    bboxes2 = boxes2.xyxy
    classes2 = boxes2.cls.astype(np.int32)
    probs2 = (boxes2.conf*100).astype(np.uint8)

    return bboxes1, classes1, probs1, bboxes2, classes2, probs2
   
def PieceDetection(bboxes, classes):
    mask = (classes != 13) & (classes != 11)
    indx = np.where(mask)[0]

    return np.int0(bboxes[indx]), classes[indx]

def CheckerboardFrame(frame, bboxes, classes):
    indx = np.where(classes == 11)[0]
    ch_coords = np.int0(bboxes[indx][0])
    
    if ch_coords.shape == 0:
        ch_frame = np.array([])
    else:
        ch_frame = frame[ch_coords[1]:ch_coords[3],ch_coords[0]:ch_coords[2]]

    return ch_frame, ch_coords

def concatStreamwcoord(handflag1,handflag2, piece_classes1, piece_classes2, piece_bboxes1,piece_bboxes2, probs1, probs2, sorted_coord1, sorted_coord2):
    stream = []
    stream.append(handflag1)
    stream.append(handflag2)

    detFlag = np.sum(sorted_coord1) > 0 and np.sum(sorted_coord2) > 0
    stream.append(int(detFlag))
    
    stream.append(len(piece_classes1.tolist()))
    stream.append(len(piece_classes2.tolist()))

    stream.extend(piece_classes1.tolist())
    stream.extend(piece_classes2.tolist())
    
    stream.extend(piece_bboxes1.reshape(-1).tolist())
    stream.extend(piece_bboxes2.reshape(-1).tolist())

    stream.extend(probs1.tolist())
    stream.extend(probs2.tolist())

    if detFlag:
        stream.extend(sorted_coord1.reshape(-1).tolist())
        stream.extend(sorted_coord2.reshape(-1).tolist())
        global ch_flag 
        ch_flag = False

    return stream

def concatStreamwcoord_nocoord(handflag1,handflag2, piece_classes1, piece_classes2, piece_bboxes1,piece_bboxes2, probs1, probs2):
    stream = []
    stream.append(handflag1)
    stream.append(handflag2)
    
    stream.append(0)

    stream.append(len(piece_classes1.tolist()))
    stream.append(len(piece_classes2.tolist()))

    stream.extend(piece_classes1.tolist())
    stream.extend(piece_classes2.tolist())
    
    stream.extend(piece_bboxes1.reshape(-1).tolist())
    stream.extend(piece_bboxes2.reshape(-1).tolist())

    stream.extend(probs1.tolist())
    stream.extend(probs2.tolist())


    return stream


######## YOLOfuncs.py END ########



if __name__ == "__main__":
    __main__()


