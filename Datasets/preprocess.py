import numpy as np
import cv2
import os
from Datasets.gtParser import gtParser
import re

class Preprocess():
    def __init__(self):
        pass

    def crop(self, xdiv, ydiv, img, bBoxes=None):
        """
        Splits a ndarray in xdiv * ydiv similiar-sized crops.
        """
        xstride = img.shape[1] // xdiv
        ystride = img.shape[0] // ydiv

        widthLimits = np.zeros((xdiv+1,), dtype=np.int32)
        heightLimits = np.zeros((ydiv+1), dtype=np.int32)
        croppedImages = [[] for _ in range(xdiv*ydiv)]
        croppedBoxes = [[] for _ in range(xdiv*ydiv)]
        index = 0
        for x in range(0, img.shape[1]+1, xstride):
            widthLimits[index] = x
            index += 1
        index = 0
        for y in range(0, img.shape[0]+1, ystride):
            heightLimits[index] = y
            index+=1
        index = 0
        for i in range(len(widthLimits)-1):
            for j in range(len(heightLimits)-1):
               croppedImages[index] = img[heightLimits[j]:heightLimits[j+1], widthLimits[i]:widthLimits[i+1]]
               index += 1
        if bBoxes:
            for box in bBoxes:
                index = 0
                for i in range(len(widthLimits)-1):
                    for j in range(len(heightLimits)-1):
                        if box[0] >= widthLimits[i] and box[2] < widthLimits[i+1] \
                        and box[1] >= heightLimits[j] and box[3] < heightLimits[j+1]:
                            box[0] -= widthLimits[i]
                            box[2] -= widthLimits[i]
                            box[1] -= heightLimits[j]
                            box[3] -= heightLimits[j]
                            croppedBoxes[index].append(box)
                        index += 1
        return croppedImages, croppedBoxes

    def extractVirusSignal(self, srcDir):
       
        parser = gtParser('Pamonodaten')
        gtDict = parser.parse(index=True)
        files = os.listdir(srcDir)
        for i in range(len(files)-1, -1, -1):
            if not files[i].endswith('.png'):
                files.pop(i)
        imageShape = cv2.imread(os.path.join(srcDir, files[0]), cv2.IMREAD_ANYDEPTH).shape
        slwImage = 40
        slwBackground = 40
        gap = 40
        virusSignals = {}
        #get current image approximation:
        
        index = 1
        for key in gtDict:
            # For debugging purposes
            if index == 2:
                pass
            ########################
            currImageFrames = np.zeros((slwImage, imageShape[0], imageShape[1]), dtype=np.uint16)
            currImageNames = []
            backgroundImageFrames = np.zeros((slwBackground, imageShape[0], imageShape[1]), dtype=np.uint16)
            backgroundNames = []
            imageApprox = np.zeros(imageShape, dtype=np.uint16)
            backgroundApprox = np.zeros(imageShape, dtype=np.uint16)
            frameNumber = int(key)
            currImgStart = frameNumber + gap//2 
            bgImgStart = frameNumber - gap//2 - slwBackground

            for tc in range(slwImage):
                currImageFrames[tc] = cv2.imread(os.path.join(srcDir, files[min(currImgStart+tc, len(files)-1)]), cv2.IMREAD_ANYDEPTH)  
            for tb in range(slwBackground):
                backgroundImageFrames[tb] = cv2.imread(os.path.join(srcDir, files[max(0, bgImgStart+tb)]), cv2.IMREAD_ANYDEPTH)
            
            currImageFrames = np.transpose(currImageFrames, (1, 2, 0))
            backgroundImageFrames = np.transpose(backgroundImageFrames, (1, 2, 0))
            for x in range(imageShape[1]):
                for y in range(imageShape[0]):
                    imageApprox[y, x] = np.median(np.sort(currImageFrames[y, x], axis=None))
                    backgroundApprox[y, x] = np.median(np.sort(backgroundImageFrames[y, x], axis=None))

            epsilon = 1 / (2**-16 - 1)
            VirusSignal = np.array((imageApprox + epsilon) / (backgroundApprox + epsilon), dtype=np.float32)
           
            virusSignals[frameNumber] = VirusSignal
            print('Read Image {}/{} ...'.format(index, len(gtDict)))
            index += 1
        return virusSignals

    def saveVirusSignals(self, virusSignals: dict, srcDir):
        parser = gtParser('Pamonodaten')
        gtDict = parser.parse(index=True)
        path = os.path.join(srcDir, 'preprocessed')
        if not os.path.exists(path):
            os.mkdir(path)
        index = 1

        for key in virusSignals:
            image = virusSignals[key]
            # Normalie images to 0-65535 range
            image = cv2.normalize(image, dst=None, alpha=0, beta=65535,norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_16UC1)
            #image = self.increaseContrast(image)
            #
            # image = cv2.GaussianBlur(image, (0, 0), 1)
            # boxes = gtDict[str(key)]
            # for gtBox in boxes:
            #      cv2.rectangle(image, (gtBox[0], gtBox[1]), (gtBox[2], gtBox[3]), (255,0,0))
            if int(key) < 10:
                namePattern = '80nm000{}.png'.format(key)
            elif int(key) < 100:
                namePattern = '80nm00{}.png'.format(key) 
            elif int(key) < 1000:
                namePattern = '80nm0{}.png'.format(key)
            else:
                namePattern = '80nm{}.png'.format(key)
            cv2.imwrite(os.path.join(srcDir, 'preprocessed', namePattern), image)
            print('Saved image {}/{}'.format(index, len(virusSignals)))
            index += 1

    def increaseContrast(self, image):
        alpha = 1.2
        contrastImage = np.array(image.copy(), dtype=np.float64)
        mean = np.mean(contrastImage)
        contrastImage = np.multiply(contrastImage, alpha)
        contrastImage[contrastImage > 65535] = 65535
        contrastImage = np.array(contrastImage, dtype=np.uint16)
        contrastImage = cv2.GaussianBlur(contrastImage, (3, 3), 1)
        
        return contrastImage

    def colorCorrection(self, srcDir):
        files = os.listdir(srcDir)
        for i in range(len(files)-1, -1, -1):
            if not files[i].endswith('.png'):
                files.pop(i)
            elif files[i].endswith('color.png'):
                files.pop(i)
        firstImage = cv2.imread(os.path.join(srcDir, files[0]), cv2.IMREAD_ANYDEPTH)
        globalBrightness = np.mean(firstImage)
        index = 1
        for imagePath in files:
            print('Editing image {}/ {}'.format(index, len(files)))
            image = cv2.imread(os.path.join(srcDir, imagePath), cv2.IMREAD_ANYDEPTH)
            max = np.max(image)
            min = np.min(image)
            localBrightness = np.mean(image)
            brightnessFactor = globalBrightness / localBrightness
            image = np.array(np.multiply(image, brightnessFactor), dtype=np.int32)
            debug = image[image > 65535]
            image[image > 65535] = 65535
            nmax = np.max(image)
            nmin = np.min(image)
            image = np.array(image, dtype=np.uint16)
            cv2.imwrite(os.path.join(srcDir, imagePath), image)
            index += 1
        print('finished')

    def extractAndSave(self, srcDir):
        virusSignals = self.extractVirusSignal(srcDir)
        self.saveVirusSignals(virusSignals, srcDir)


    def getPixelMean(self, xy, slidingWindowFrames):
        pixelValues = np.zeros(len(slidingWindowFrames), dtype=np.uint16)
        for i in range(len(slidingWindowFrames)):
            pixelValues[i] = slidingWindowFrames[i][xy[1], xy[0]]
        pixelValues = np.sort(pixelValues, axis=None)
        median = np.median(pixelValues)
        return median
        
        
if __name__ == "__main__":
    srcDir = os.path.join('Pamonodaten', '80and200 27July2016')
    test = Preprocess()
    test.extractAndSave(srcDir)
       