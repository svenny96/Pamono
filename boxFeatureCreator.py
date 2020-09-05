from Datasets.gtParser import gtParser
import sys
import numpy as np
from miscUtils import filterImageFiles, calculateIou
import os
import cv2 as cv
import shapely
from shapely.geometry import Point, Polygon
import csv
import math

FBOXSIZE = 4000

class boxFeature():
    def __init__(self, subDir, windowLength):
        parser = gtParser('Pamonodaten')
        self.groundTruth = parser.parseSingle(subDir, index=True)
        self.minSize, self.maxSize = self.__stats()
        self.imageFiles = filterImageFiles(os.path.join('Pamonodaten', subDir))
        self.shape = cv.imread(os.path.join('Pamonodaten', subDir, self.imageFiles[0]), cv.IMREAD_ANYDEPTH).shape
        print(self.shape)
        self.windowLength = windowLength
        self.subDir = subDir
        self.anchorBoxes = self.__generateAnchorBoxes()

    def __stats(self):
        minSize = 1000
        maxSize = 0
        for key in self.groundTruth:
            for box in self.groundTruth[key]:
                size = (box[2]-box[0])*(box[3]-box[1])
                if size < minSize:
                    minSize = size
                if size > maxSize:
                    maxSize = size
        return minSize, maxSize

    def containsOverlap(self, singleBox, gtBoxList):
        for box in gtBoxList:
            if calculateIou(singleBox, box) > 0.3:
                return True
        return False

    def loadFrames(self):
        print('loading frames...')
        start = 0
        end = 0
        for key in self.groundTruth:
            if int(key) > end:
                end = int(key)
        end += self.windowLength//2+1
        frames = np.zeros((end, self.shape[0], self.shape[1]), dtype=np.uint16)
        for i in range(min(end, len(self.imageFiles))):
            frames[i,:,:] = cv.imread(os.path.join('Pamonodaten', self.subDir, self.imageFiles[i]), cv.IMREAD_ANYDEPTH) # increment end by half window length
        print('finished loading.')
        return frames

    def __generateAnchorBoxes(self):
        width = math.ceil(math.sqrt(self.maxSize))
        backgroundBoxes = []
        for x in range(0, self.shape[1], width):
            for y in range(0, self.shape[0], width):
                slidingBox = [x, y, x+width, y+width]
                backgroundBoxes.append(slidingBox)
        return backgroundBoxes

    def randomizeBoxSize(self, negativeBoxes):
        for i in range(len(negativeBoxes)):
            length = np.random.randint(self.minSize, self.maxSize) 
            negativeBoxes[i][:,length:] = 0

    def saveTrainingData(self, posBoxes, negBoxes, maxEntries=300):
        fileName = 'lstmGroundTruthv2.csv'
        counter = 0
        with open(os.path.join('Pamonodaten', self.subDir, fileName),'w', newline='') as csvFile:
            writer = csv.writer(csvFile)
            for box in posBoxes:
                if counter == maxEntries:
                    break
                labeledBox = np.zeros((box.shape[0]+1), dtype=np.uint16)
                labeledBox[0:box.shape[0]] = box
                labeledBox[-1] = 1
                np.savetxt(csvFile, [labeledBox], delimiter=';', newline='\n', fmt='%i')
                counter += 1
            counter = 0
            for box in negBoxes:
                if counter == maxEntries:
                    break
                labeledBox = np.zeros((box.shape[0]+1), dtype=np.uint16)
                labeledBox[0:box.shape[0]] = box
                np.savetxt(csvFile, [labeledBox], delimiter=';', newline='\n', fmt='%i')
                counter += 1

    def generateFeatureBoxes(self):
        frames = self.loadFrames()
        virusBoxes = []
        backgroundBoxes = []
        for key in self.groundTruth:
            gtBoxList = []
            bgBoxList = []
            # Collect all ground-truth boxes at frame index key:
            for box in self.groundTruth[key]:
                gtBoxList.append(box)
            # Eliminate background Boxes which overlap with any ground-truth box
            localBackgroundBoxes = [box for box in self.anchorBoxes if box[2] < self.shape[1] and box[3] < self.shape[0] and not self.containsOverlap(box, gtBoxList)]
            # Choose as many random boxes as gtBoxes randomly
            randomIndeces = np.random.choice(len(localBackgroundBoxes), len(gtBoxList),replace=False)
            for index in randomIndeces:
                bgBoxList.append(localBackgroundBoxes[index])

            for box in gtBoxList:
                if int(key) > frames.shape[0]-self.windowLength//2:
                    imageCrop = frames[frames.shape[0]-self.windowLength:frames.shape[0],box[1]:box[3],box[0]:box[2]]
                elif int(key) < self.windowLength//2:
                    imageCrop = frames[0:self.windowLength,box[1]:box[3],box[0]:box[2]] 
                else:
                    imageCrop = frames[max(0,int(key)-self.windowLength//2):int(key)+self.windowLength//2+1,box[1]:box[3],box[0]:box[2]]
                imageCrop =  imageCrop.reshape(imageCrop.shape[0], imageCrop.shape[1]*imageCrop.shape[2])
                #featureBox = np.zeros((self.windowLength+1,FBOXSIZE), dtype=np.uint16)
                #featureBox[:,0:imageCrop.shape[1]] = imageCrop
                featureBox = np.average(imageCrop, axis=1)
                virusBoxes.append(featureBox)
            for box in bgBoxList:    
                if int(key) > frames.shape[0]-self.windowLength//2:
                    imageCrop = frames[frames.shape[0]-self.windowLength:frames.shape[0],box[1]:box[3],box[0]:box[2]]
                elif int(key) < self.windowLength//2:
                    imageCrop = frames[0:self.windowLength,box[1]:box[3],box[0]:box[2]] 
                else:
                    imageCrop = frames[max(0,int(key)-self.windowLength//2):int(key)+self.windowLength//2+1,box[1]:box[3],box[0]:box[2]]
                imageCrop =  imageCrop.reshape(imageCrop.shape[0], imageCrop.shape[1]*imageCrop.shape[2])
                #featureBox = np.zeros((self.windowLength+1,FBOXSIZE), dtype=np.uint16)
                #featureBox[:,0:imageCrop.shape[1]] = imageCrop
                featureBox = np.average(imageCrop, axis=1)
                backgroundBoxes.append(featureBox)

        #self.randomizeBoxSize(backgroundBoxes)
        self.saveTrainingData(virusBoxes, backgroundBoxes)

if __name__ == "__main__":
    bf = boxFeature('PP_BG_eliminated', 100)
    print(bf.generateFeatureBoxes())