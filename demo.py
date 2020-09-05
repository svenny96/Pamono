import torch
import CustomRCNN
import LSTM
from Datasets.RCNNDataset import RCNNDataset
from Datasets.LSTMDataset import LSTMDataset
import torchvision.transforms.functional as F
from PIL import Image, ImageDraw, ImageOps
from torchvision import transforms
import os
from pathlib import Path
import csv
import numpy
import cv2
from torchvision import transforms
from PIL import Image, ImageDraw, ImageOps
from Datasets.preprocess import Preprocess
from Datasets.gtParser import gtParser
import miscUtils as misc
import torch.nn.functional as F
from functools import reduce
from sklearn import metrics
from matplotlib import pyplot
from collections import OrderedDict

class Demo():
    def __init__(self, stateDictPathRCNN, stateDictPathLSTM):
        self.stateDictPathRCNN = stateDictPathRCNN
        self.stateDictPathLSTM = stateDictPathLSTM
        self.RCNNModel = self.__loadRCNNModel(stateDictPathRCNN)
        self.LSTMModel = self.__loadLSTMModel(stateDictPathLSTM)
        self.topDir = 'Pamonodaten'

    def writeCsv(self, imgDir, picNames, scores):
        with open(imgDir + '\outputTensor.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['index', 'picName', 'clsScores'])
            for i in range(len(picNames)):
                with open(imgDir + '\outputTensor.csv', 'a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow([i, picNames[i], scores[i]])

    def __loadRCNNModel(self, stateDictPath='Results\epoch1model.pth'):
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        model = CustomRCNN.getCustomModel()
        model.load_state_dict(torch.load(os.path.join('Results', stateDictPath)))
        model.to(device)
        model.eval()
        return model

    def __loadLSTMModel(self, stateDictPath):
        model = LSTM.getLSTMModel()
        model.load_state_dict(torch.load(os.path.join('Results', stateDictPath)))
        model.eval()
        return model

    def averageIou(self, gtDict, predBoxes):
        iouSum = []
        for key in gtDict:
            for gtBox in gtDict[key]:
                ious = numpy.zeros((len(predBoxes[int(key)])))
                for i in range(len(predBoxes[int(key)])):
                    ious[i] = misc.calculateIou(predBoxes[int(key)][i], gtBox)
                if ious.shape[0] == 0:
                    iouSum.append(0)
                else:
                    iouSum.append(numpy.amax(ious))
        return reduce(lambda a, b : a+b, iouSum) / len(iouSum)

    def containsOverlap(self, boxList, singleBox, iouThresh=0.75):
        for box in boxList:
            if misc.calculateIou(box, singleBox) >= iouThresh:
                return True
        return False
                    
    def drawResults(self, imageDir, lstm=False, lstmWindow = 41):
        parser = gtParser(self.topDir)
        gtDict = parser.parseSingle(imageDir, index=True)
        imageFiles = misc.filterImageFiles(os.path.join(self.topDir, imageDir))
        shape = cv2.imread(os.path.join('Pamonodaten', imageDir, imageFiles[0]), cv2.IMREAD_ANYDEPTH).shape
        subDir = list(gtDict)[0].split('\\')[0]   
        predBoxesList = []
        predScoresList = []
        frames = numpy.zeros((len(imageFiles), shape[0], shape[1]))
        for i in range(len(imageFiles)):
            frames[i] = cv2.imread(os.path.join('Pamonodaten', imageDir, imageFiles[i]), cv2.IMREAD_ANYDEPTH)
        #self.histIntensity(frames,700,55)
        #return
        lastFrameIndex = frames.shape[0]
        printCounter = 0
        for file in imageFiles:
            image = cv2.imread(os.path.join('Pamonodaten',imageDir, file), cv2.IMREAD_ANYDEPTH)
            index = imageFiles.index(file)
            predBoxes,predScores = self.predict(image, xdiv=1, ydiv=1)     # Subject to change depending on cropping
            image = numpy.array(numpy.true_divide(image, 255), dtype=numpy.uint8)
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)                 
            # check if list index is already populated
            if len(predBoxesList) == index:
                predBoxesList.append([])
                predScoresList.append([])
            for box in predBoxes:
                predBoxesList[index].append(box)
            for score in predScores:
                predScoresList[index].append(score)
            print('Predicting image [{}/{}]'.format(printCounter, len(imageFiles)))
            printCounter += 1
            # debug
            # if printCounter == 100:
            #     break0
        # if lstm:
        #     lstmPreds = self.lstmFilter(predBoxesList, frames)
        #self.evaluate(gtDict, predBoxesList,lstmPreds,imageDir)
        print(len(predBoxesList))
        for j in range(len(predBoxesList)):
            image = cv2.imread(os.path.join('Pamonodaten', imageDir, imageFiles[j]), cv2.IMREAD_ANYDEPTH)
            image = numpy.array(numpy.true_divide(image, 256), dtype=numpy.uint8)
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            if str(j) in gtDict:
                gtBoxes = gtDict[str(j)]
                for box in gtBoxes:
                    #testBox = frames[j-20:j+20+1, int(box[1]):int(box[3]), int(box[0]):int(box[2])]
                    #featureBox = numpy.zeros((testBox.shape[0], 4000))
                    #testBox = testBox.reshape(testBox.shape[0], testBox.shape[1]*testBox.shape[2])
                    #featureBox[:,0:testBox.shape[1]] = testBox
                    #print(self.lstmPredict(self.LSTMModel, featureBox))
                    cv2.rectangle(image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0,255,0))
            # elif str(j) in filteredBoxes:
            #     print('drawingBoxes')
            #     removedBoxes = filteredBoxes[str(j)]
            #     for box in removedBoxes:
            #         cv2.rectangle(image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0,0,255))
            for box in predBoxesList[j]:
                if j > 0:
                    if self.containsOverlap(predBoxesList[j-1], box):
                        cv2.rectangle(image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0,0,255))
                    else:
                        cv2.rectangle(image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255,0,0))
                       
            cv2.imwrite(os.path.join('Results', 'Visualized', imageDir+imageFiles[j]), image) 


    def lstmFilter(self, predBoxesList: list, frames):
        # 3d input tensor (timeSteps x batches x featureSize)
        lstmModel = self.LSTMModel
        lstmModel.eval()
        filteredBoxes = {}
        lstmPred = []
        for i in range(len(predBoxesList)):
            boxes = predBoxesList[i]
            lstmPred.append([])
            removedBoxes = []
            for j in range(len(boxes)):
                box = boxes[j]
                if i < 100:
                    gtBoxCrop = frames[0:101,int(box[1]):int(box[3]), int(box[0]):int(box[2])]
                elif i > len(predBoxesList)-100:
                    gtBoxCrop = frames[len(predBoxesList)-101:len(predBoxesList), int(box[1]):int(box[3]), int(box[0]):int(box[2])]
                else:
                    gtBoxCrop = frames[i-50:i+51, int(box[1]):int(box[3]), int(box[0]):int(box[2])]
                #featureBox = numpy.zeros((gtBoxCrop.shape[0], 4000))
                gtBoxCrop = gtBoxCrop.reshape(gtBoxCrop.shape[0], gtBoxCrop.shape[1]*gtBoxCrop.shape[2]) 
                featureBox = numpy.average(gtBoxCrop,axis=1)
                logit = torch.sigmoid(self.lstmPredict(lstmModel, featureBox)).item() 
                lstmPred[i].append(logit)
                
            filteredBoxes[str(i)] = removedBoxes
            print('filtered boxes {}/{}'.format(i,len(predBoxesList)))
        return lstmPred

    def lstmPredict(self, model, featureBox):
        featureBox = numpy.divide(featureBox, 65536, dtype=numpy.float32)
        std = numpy.std(featureBox)
        mean = numpy.mean(featureBox)
        for i in range(featureBox.shape[0]):
            featureBox[i] = (featureBox[i] - mean ) / std
        lstmInput = torch.from_numpy(featureBox).unsqueeze(1).unsqueeze(2)
        #print(lstmInput.size())
        model.hidden = model.init_hidden(1)     
        logit = model(lstmInput)
        return logit

    def predict(self, image, xdiv=1, ydiv=1, scoreThresh=0):  
        prep = Preprocess()
        tensorList = []
        tensorImage = numpy.array(numpy.divide(image, 65535), dtype=numpy.float32) 
        tensorImage = torch.from_numpy(tensorImage)
        tensorList.append(tensorImage.unsqueeze(0).cuda())
        output = self.RCNNModel(tensorList)
        boxes = []
        for k in range(len(output)):
            predBoxes = output[k]['boxes'].detach().cpu().numpy()
            predScores = output[k]['scores'].detach().cpu().numpy()
            predBoxes = predBoxes[predScores >= scoreThresh]
            boxes.append(predBoxes)
        
        return predBoxes, predScores
    
    def evaluate(self, gtBoxes, predBoxes, predScores,imageDir, iouThresh=0.1):
        metricList = []
        sumGtBoxes = 0
        for key in gtBoxes:
            currentGtBoxes = gtBoxes[key]
            if int(key) >= len(predBoxes):
                break
            sumGtBoxes += len(currentGtBoxes)
            currentPredBoxes = predBoxes[int(key)]
            currentScores = predScores[int(key)]
            # test predicted Boxes for IoU with gt Boxes
            assignedGtBoxes = numpy.zeros((len(currentGtBoxes)))
            for i in range(len(currentPredBoxes)):
                predBoxIous = numpy.zeros((len(currentGtBoxes)))
                
                for j in range(len(currentGtBoxes)):
                    predBoxIous[j] = misc.calculateIou(currentPredBoxes[i], currentGtBoxes[j])
                maxIou = numpy.amax(predBoxIous)
                gtBoxToAssign = numpy.argmax(predBoxIous)
                if maxIou >= iouThresh and assignedGtBoxes[gtBoxToAssign] == 0:
                    metricList.append({'score':currentScores[i],'iou':True})
                    assignedGtBoxes[gtBoxToAssign] = 1
                else:
                    metricList.append({'score':currentScores[i],'iou':False})
        sortedMetrics = sorted(metricList, key= lambda dict : dict['score'],reverse=True)
        precision = []
        recall = []
        falseNegatives = 0
        truePositives = 0
        falsePositives = 0

        with open(os.path.join('Pamonodaten', 'precisionRecall.csv'),'w', newline='') as csvFile:
            writer = csv.writer(csvFile,delimiter=';')
            writer.writerow(['precision','recall','thresh'])

            for i in range(len(sortedMetrics)):
                if sortedMetrics[i]['iou'] == True:
                    truePositives += 1
                else:
                    falsePositives += 1
                # Number format for Ms Excel
                precision = truePositives / (truePositives+falsePositives)
                precision = str(precision).replace('.',',')
                recall = str(truePositives / sumGtBoxes).replace('.',',')
                score = str(sortedMetrics[i]['score']).replace('.',',')

                writer.writerow([precision,recall,score])
        
        print('hello')

       
    def evaluateLstm(self,subDir):
        model = self.LSTMModel
        lstmDataset = LSTMDataset(subDir, boxFeature=True)
        pred = []
        targets = []
        dataloader = torch.utils.data.DataLoader(lstmDataset, batch_size=1, num_workers=1, shuffle=False)
        for i, item in enumerate(dataloader):
            input = item['sequence'].unsqueeze(2).permute(1,0,2)
            target = item['target'].unsqueeze(1)
            model.hidden = model.init_hidden(1)
            output = torch.sigmoid(model(input)) 
            pred.append(output.item())
            targets.append(target.item())
            print(i)

        fpr, tpr, threshholds = metrics.roc_curve(targets,pred)
        areaUnderCurve = metrics.auc([0,0.5,1],[0,0.5,1])
        print('AUC: {}'.format(areaUnderCurve))
        with open(os.path.join('Pamonodaten', 'lstmROC.csv'),'w', newline='') as csvFile:
            writer = csv.writer(csvFile)
            writer.writerow(['fpr','tpr','thresh'])
            for i in range(len(fpr)):
                writer.writerow(['{}'.format(fpr[i]),'{}'.format(tpr[i]),'{}'.format(threshholds[i])])
        print('done')

        
    def histIntensity(self, frames, x, y):  
        for i in range(frames.shape[0]):
            print(frames[i,y,x])

    def intersectionOverlap(self,singleGtBox, predBoxes):
        gtArea = (singleGtBox[2] - singleGtBox[0]) * (singleGtBox[3] - singleGtBox[1])
        for box in predBoxes:
            predArea = (box[2] - box[0]) * (box[3] - box[1])
            intersection = misc.rectIntersection(singleGtBox, box)
            if intersection / gtArea >= 0.75 or intersection / predArea >= 0.75:
                return True
        return False

    def singleIntersectionOverlap(self, boxA, boxB):
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        intersection = misc.rectIntersection(boxA, boxB)
        if intersection / boxAArea >= 0.75 or intersection / boxBArea >= 0.75:
            return True
        else:
            return False


class LSTMDemo():
    def __init__(self, stateDictName, subDir):
        self.stateDictPath = os.path.join('Results', stateDictName)
        self.LSTMModel = LSTM.getLSTMModel() 
        self.subDir = subDir

    def showPixelMasks(self):
        parser = gtParser('Pamonodaten')
        # LSTM needs context of frame-window
        gtDict = parser.parseSingle(self.subDir)
        model = self.LSTMModel
        model.load_state_dict(torch.load(self.stateDictPath))
        model.eval()
        # 3d input tensor (timeSteps x batches x featureSize)
        images = self.extractWindow(41, 30)
        pixelMask = numpy.zeros((images.shape[0], images.shape[1]))
        progress = 0
        for x in range(images.shape[1]):
            for y in range(images.shape[0]):
                model.hidden = model.init_hidden(1)
                model.zero_grad()
                # normalize input sequence
                inputSequence = numpy.divide(images[y,x,:], 65535) 
                inputSequence = torch.from_numpy(inputSequence).unsqueeze(1).unsqueeze(2)
                output = model(inputSequence)
                # assign 1 to virus pixels
                if output >= 0:
                    pixelMask[y,x] = 1
                print('Pixel {}'.format(progress))
                progress += 1

        indexedImage = numpy.asarray(numpy.median(images, axis=2), dtype=numpy.uint16)
        # convert from grayscale to BGR for colored mask highlighting
        indexedImage = cv2.cvtColor(indexedImage, cv2.COLOR_GRAY2BGR) 
        indexedImage[numpy.nonzero(pixelMask)] = (65535, 0, 0)
        cv2.imshow('test', indexedImage)
        cv2.waitKey(0)
        cv2.destroyAllWindows() 

    def extractWindow(self, sequenceLength, imageIndex):
        imageFiles = misc.filterImageFiles(os.path.join('Pamonodaten', self.subDir))
        shape = cv2.imread(os.path.join('Pamonodaten', self.subDir, imageFiles[0]), cv2.IMREAD_ANYDEPTH).shape
        window = numpy.zeros((shape[0],shape[1],sequenceLength), dtype=numpy.float32)

        if (imageIndex - int(sequenceLength)//2) < 0:
            start = 0
            stop = sequenceLength+1
        else: 
            start = imageIndex - int(sequenceLength//2)
            stop = imageIndex + int(sequenceLength//2)+1
        index = 0
        for i in range(start, stop):
            window[:,:,index] = cv2.imread(os.path.join('Pamonodaten', self.subDir, imageFiles[i]), cv2.IMREAD_ANYDEPTH)
            index += 1
        print(window.shape)
        return window

if __name__ == "__main__":
    demo = Demo('final.pth', 'avgBrightness2LSTM.pth')      
    demo.drawResults('200nm_9May14_cropped', lstm=False)
    #demo.evaluateLstm(['200nm_11Apr13_1'])
    #lstmDemo = LSTMDemo('testLSTM.pth', 'PP_BG_eliminated')
    #lstmDemo.showPixelMasks()