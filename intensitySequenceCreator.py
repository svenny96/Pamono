from Datasets.gtParser import gtParser
import sys
import numpy as np
from miscUtils import filterImageFiles
import os
import cv2 as cv
import shapely
from shapely.geometry import Point, Polygon
import csv

OUTPUTLENGTH = 300

def loadFrames(start, end):
    """
    Return buffer containing images along a given time-span
    """
    #end += 40
    frames = np.zeros((shape[0], shape[1], end), dtype=np.uint16)
    for i in range(end):
        frames[:,:,i] = cv.imread(os.path.join('Pamonodaten', dirName, imageFiles[i]), cv.IMREAD_ANYDEPTH)
    return frames

def numpyToShapely(polygon):
    shapelyCoords = []
    for i in range(polygon.shape[0]):
        point = (polygon[i,0], polygon[i,1])
        shapelyCoords.append(point)
    return shapelyCoords

def clipSequence(sequence):
    max = np.argmax(sequence)
    if max-OUTPUTLENGTH//2 < 0:
        return sequence[0:OUTPUTLENGTH]
    elif max+int(OUTPUTLENGTH/2+1) >= sequence.shape[0]:
        return sequence[-OUTPUTLENGTH-1:-1]
    if OUTPUTLENGTH % 2 == 0:
        return sequence[max-OUTPUTLENGTH//2:max+OUTPUTLENGTH//2]
    else:
        return sequence[max-int(OUTPUTLENGTH//2):max+int(OUTPUTLENGTH/2+1)]

dirName = str(sys.argv[1])
parser = gtParser('Pamonodaten')
dict = parser.parseSingle(dirName, index=True, polygon=True)
start = 99999
end = 0
gtPolygons = []
for key in dict:
    if int(key) < start:
        start = int(key)
    elif int(key) > end:
        end = int(key) 
    for array in dict[key]:
        gtPolygons.append(array)
print('Start', start, 'Ende', end)
imageFiles = filterImageFiles(os.path.join('Pamonodaten', dirName))
shape = cv.imread(os.path.join('Pamonodaten', dirName, imageFiles[0]), cv.IMREAD_ANYDEPTH).shape
frames = loadFrames(start, end)
pixelLabels = np.zeros(shape, dtype=np.int8)

# Create a list of tuples (x, y, label) for each pixel 
for polygon in gtPolygons:
    polygon = np.reshape(polygon, (-1,2))
    minX = int(round(np.amin(polygon, axis=0)[0]))
    minY = int(round(np.amin(polygon, axis=0)[1]))
    maxX = int(round(np.amax(polygon, axis=0)[0]))
    maxY = int(round(np.amax(polygon, axis=0)[1]))
    shapelyPoly = Polygon(numpyToShapely(polygon))
    for x in range(minX, maxX+1):
        for y in range(minY, maxY+1):
            shapelyPoint = Point(x, y)
            if shapelyPoint.within(shapelyPoly):
                pixelLabels[y-1,x-1] = 1
            else:
                pixelLabels[y-1,x-1] = -1
print(pixelLabels)
virusSeqs = []
backgroundSeqs = []
print('extract sequences')
fileName = 'lstmGroundTruth.csv'
virusPixels = np.argwhere(pixelLabels == 1)
backgroundPixels = np.argwhere(pixelLabels == 0)
randomSample = np.random.choice(backgroundPixels.shape[0], np.where(pixelLabels==1)[0].shape[0], False)
backgroundPixels = backgroundPixels[randomSample,:]
gtPixels = np.concatenate((virusPixels, backgroundPixels))
count = 0
with open(os.path.join('Pamonodaten', dirName, fileName),'w') as csvFile:
    for position in gtPixels:
        label = pixelLabels[position[0], position[1]]
        print('Writing {}/{}'.format(count, shape[0]*shape[1]))
        seq = np.zeros((frames.shape[2]+1), dtype=np.uint16)
        seq[0:frames.shape[2]] = frames[position[0]-1,position[1]-1,:]
        clippedSequence = np.zeros((OUTPUTLENGTH+1), dtype=np.uint16)
        clippedSequence[0:OUTPUTLENGTH] = clipSequence(seq)
        clippedSequence[-1] = label
        np.savetxt(csvFile, [clippedSequence], delimiter=';', newline='\n', fmt='%i')
            
        count += 1

        
print('finished')