import os
import re
import math

def filterImageFiles(imageDirPath):
    """
    Returns image files (.png) given a path to an image directory
    """
    files = os.listdir(imageDirPath)
    imageFiles = []
    for file in files:
        if file.endswith('png'):
            imageFiles.append(file)
    return imageFiles

def preprocessed2Index(filename):
    """
    Returns the ground-truth index of preprocessed images
    """
    indexPattern = re.compile('[0-9]*\.')
    match = indexPattern.search(filename)
    return match.group().split('.')[0]

def calculateIou(boxA, boxB):
        """
        from https://www.pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/
        """
        # determine the (x, y)-coordinates of the intersection rectangle
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
    
        # compute the area of intersection rectangle
        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    
        # compute the area of both the prediction and ground-truth
        # rectangles
        boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    
        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the interesection area
        iou = interArea / float(boxAArea + boxBArea - interArea)
    
        # return the intersection over union value
        return iou

def rectIntersection(boxA, boxB):
    leftX = max(boxA[0], boxB[0])
    rightX = min(boxA[2], boxB[2])
    bottomY = max(boxA[1], boxB[1])
    topY = min(boxA[3], boxB[3])

    if leftX < rightX and bottomY < topY:
        return (rightX - leftX) * (topY - bottomY)
    else:
        return 0