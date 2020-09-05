import os
import re
import numpy as np
from pathlib import Path
import csv

class gtParser():
    def __init__(self, topDir):
        self.topDir = topDir

    def __findCsv(self, subDir):
        subpath = os.path.join(self.topDir, subDir)
        files = os.listdir(subpath)
        for fileName in files:
            if fileName.endswith('.csv'): 
                return os.path.join(self.topDir,subDir, fileName)

    def __toBoundingBox(self, xyList):
        coords = np.asarray(xyList, dtype=np.float)
        coords = np.reshape(coords, (-1, 2))

        minXY = np.nanmin(coords, 0)
        maxXY = np.nanmax(coords, 0)

        return [int(minXY[0]), int(minXY[1]), int(maxXY[0]), int(maxXY[1])]

    def __toPolygon(self, xyList):
        coords = np.reshape(np.asarray(xyList, dtype=np.float), (-1,2))
        coords = coords[np.logical_not(np.isnan(coords))]
        return coords

    def __toImageName(self, imagePath):
        namePattern = re.compile('([A-Z]|[0-9]| |_|-)+\.png', re.IGNORECASE)
        match = namePattern.search(imagePath)

        return match.group()

    def parseSingle(self, subDir, index=False, polygon=False):
        """
        Returns filenames of images and the associated ground-truth data

        Parameters:
        ----------
        subDir : String 
            Name of the directory to parse the images from. Must contain annotations as .csv
        index : Boolean
             Choose between image index (index=True) and image name(index=False) as key 
        polygon : Boolean
             True: outputs annotations as polygon coordinates False: outputs annotations as rectangular bounding box coordinates x1,y1,x2,y2
        """
        polygons = {}
        csvPath = self.__findCsv(subDir)
        
        with open(csvPath, newline='') as file:
            csvReader = csv.DictReader(file, delimiter=';')           
            for row in csvReader:
                coords = []
                if index:
                    currentName = row['frameNumber']
                else:
                    currentName = os.path.join(subDir, self.__toImageName(row['fileName']))
                if currentName not in polygons:
                    polygons[currentName] = []
                for i in range(100):
                    try:
                        coords.append(row['x{}'.format(i)])
                        coords.append(row['y{}'.format(i)])
                    except:
                        break
                if polygon:
                    polygons[currentName].append(self.__toPolygon(coords))
                else:
                    polygons[currentName].append(self.__toBoundingBox(coords))
                   
        #print(polygons)
        return polygons

    def parseAll(self):
        polygonDict = {}
        for dir in [dir for dir in os.listdir(os.path.join('Pamonodaten')) if os.path.isdir(os.path.join('Pamonodaten', dir))]:
            polygonDict.update(self.parseSingle(dir))
        return polygonDict

    def parseMultiple(self, subDirList):
        polygonDict = {}
        for dir in subDirList:
            polygonDict.update(self.parseSingle(dir))
        return polygonDict