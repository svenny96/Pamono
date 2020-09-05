from torch.utils.data import Dataset
import os
import numpy as np
from pathlib import Path
import collections
import torchvision.transforms.functional as F
import torch
import csv
from PIL import Image, ImageDraw, ImageOps
from Datasets.gtParser import gtParser
import Datasets.gtParser
from torchvision import transforms
import cv2 
import math
import numpy

class RCNNDataset(Dataset):
    def __init__(self, srcDir, transform=None,subDir=None):
        self.srcDir = srcDir
        self.subDir = subDir
        self.imageNames, self.gtBoxes,  = self.__getImages()
        self.transforms = transform
        

    def __getImages(self):
        parser = gtParser(self.srcDir)
        if self.subDir is not None:
            images = parser.parseMultiple(self.subDir)
        else:
            images = parser.parseAll()
        imageNames = []
        gtBoxes = []
        for key in images:
            imageNames.append(key)
            gtBoxes.append(images[key])
        
        nImages = []
        nGtBoxes = []
        nImageNames = []
        for i in range((len(imageNames))):
            imgPath = os.path.join(self.srcDir, imageNames[i])
            img = cv2.imread(imgPath, cv2.IMREAD_ANYDEPTH)
            boxes = gtBoxes[i]         
            nImages.append(img)
            nGtBoxes.append(boxes) 
        
        return nImages, nGtBoxes

    def __len__(self):
        return len(self.imageNames)

    def __getitem__(self, idx):
        #print(self.imageNames[idx])
        img = self.imageNames[idx]
        boxes = self.gtBoxes[idx]
        img = np.array(img, dtype=np.float32)
        img = np.divide(img, 65535)
        numBoxes = len(self.gtBoxes[idx])
        

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.ones(numBoxes, dtype=torch.int64)
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((numBoxes,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)
        return img, target


    def __toFileNames(self, imageList):
        nImageList = []
        for image in imageList:
            index = image.split("_")[1]
            if len(index) > 8:
                index = index[len(index)-8:]
            elif len(index) == 7:
                index = "0" + index
            elif len(index) == 6:
                index = "00" + index
            elif len(index) == 5:
                index = "000" + index
            index = "image_" + index
            nImageList.append(index)
       
        return nImageList

    def __split(self, imageList: list, polygons: dict):
        nPolygons = []
        nImages = []

        for imageName in imageList:
            img_path = os.path.join(self.srcDir, imageName)      
            #convert 16bit Image to 8bit greyscale :
            table=[ i/256 for i in range(65536) ]
            img = Image.open(img_path)
            img = img.point(table,'L')
            

            croppedImages, croppedPolygons = self._splitSingle(imageName, img, polygons[imageName])
            for j in range(len(croppedImages)):
                nImages.append(croppedImages[j])
                nPolygons.append(croppedPolygons[j])
        return nImages, nPolygons
        


    def _splitSingle(self, imageName, image, polygons: list):
        
        height = 816
        width = 1408
        topLeft = (0, 0, width //2, height//2)
        topRight = (width//2, 0, 0, height//2)
        bottomLeft = (0, height//2, width//2, 0)
        bottomRight = (width//2, height//2, 0, 0)

        croppedImages = []
        croppedImg = image.copy()
        croppedImages.append(ImageOps.crop(croppedImg, topLeft))
        croppedImages.append(ImageOps.crop(croppedImg, topRight))
        croppedImages.append(ImageOps.crop(croppedImg, bottomLeft))
        croppedImages.append(ImageOps.crop(croppedImg, bottomRight))

        # croppedImages[0].show()
        # croppedImages[1].show()
        # croppedImages[2].show()
        # croppedImages[3].show()
        croppedPolygons = collections.OrderedDict()
        croppedPolygons["0"] = []
        croppedPolygons["1"] = []
        croppedPolygons["2"] = []
        croppedPolygons["3"] = []

        for polygon in polygons:
            minXY = np.nanmin(polygon, 0)
            maxXY = np.nanmax(polygon, 0)
            #top-left crop
            if maxXY[0] < width//2 and maxXY[1] < height//2:    
                croppedPolygons["0"].append(polygon)     
            #top-right crop
            if minXY[0] >= width//2 and maxXY[1] < height//2:
                polygon[:,0] = polygon[:, 0] - width//2         
                croppedPolygons["1"].append(polygon)
            #bottom-left crop
            if maxXY[0] < width//2 and minXY[1] >= height//2:
                polygon[:, 1] = polygon[:, 1] - height//2
                croppedPolygons["2"].append(polygon)
            #bottom-right crop
            if minXY[0] >= width//2 and minXY[1] >= height//2:
                polygon[:, 0] = polygon[:, 0] - width//2
                polygon[:, 1] = polygon[:, 1] - height//2
                croppedPolygons["3"].append(polygon)
        #Remove crops which contain no ground truth polygon
        toRemove = []
        tempCroppedPolygons = []
        for key in croppedPolygons:
            if not croppedPolygons[key]:
                toRemove.append(int(key))
            else:
                tempCroppedPolygons.append(croppedPolygons[key])
        toRemove.sort(reverse=True)
        for i in range(len(toRemove)):
            croppedImages.pop(toRemove[i])
        return croppedImages, tempCroppedPolygons                      
    
    def matchingTest(self):
        randomNumber = np.random.randint(0, len(self.imageNames))
        print('Random: ', randomNumber)
        #img = cv2.imread(os.path.join(self.srcDir, self.imageNames[randomNumber]), cv2.IMREAD_ANYDEPTH)
        img = self.imageNames[randomNumber]
        cropImg = img.copy()
        gtBoxes = self.gtBoxes[randomNumber]
        imgClr = numpy.array(numpy.true_divide(img, 256), dtype=numpy.uint8)
        imgClr = cv2.cvtColor(imgClr, cv2.COLOR_GRAY2BGR)
        """  scales = [32]
        ratios=[1]
        for x in range(16,1408,32):
            for y in range(16,816,32):
                for scale in scales:
                    for ratio in ratios:
                        height = (math.sqrt(ratio))*scale
                        width = (1 / math.sqrt(ratio))*scale
                        rectangle = [int(x-width/2),int(y-height/2),int(x+width/2),int(y+height/2)]
                        cv2.rectangle(imgClr, (rectangle[0],rectangle[1]),(rectangle[2],rectangle[3]), (0,255,0),thickness=1)
        cv2.rectangle(imgClr, (x-2,y-2),(x+2,y+2), (0,0,255),thickness=-1) """
        for box in gtBoxes:
            cv2.rectangle(imgClr, (box[0], box[1]), (box[2], box[3]), (255,0,0) )
        cv2.imshow('original', imgClr)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def __contains(self, origArray, croppedArray):
        for start in range(origArray.shape[1]):
            for end in range(start, origArray.shape[1]+1):
                slice = origArray[:,start:end:1]
                if np.array_equal(slice, croppedArray):
                    return start, end
        return 0, 0

if __name__ == "__main__":
    test = RCNNDataset(Path("Pamonodaten"),subDir=['200nm_10Apr13 (complete)'])
    print(test[40][0].shape)
    test.matchingTest()
    # boxes = targets["boxes"]
    # rimg = img.copy()
    # imgDraw = ImageDraw.Draw(rimg)
    # for i in range(boxes.shape[0]):
    #     imgDraw.rectangle((boxes[i, 0], boxes[i,1], boxes[i, 2], boxes[i, 3]), outline=(255))
    # rimg.show()
    # print(np.asarray(rimg))
   
   


    
     
