
from torch.utils.data import Dataset
import os
import csv
import numpy as np

class LSTMDataset(Dataset):
    def __init__(self, subDir, boxFeature = False):
        if boxFeature:
            parser = lstmBoxParser()
            self.sequences = parser.parseMultiple(subDir)

        else:
            parser = lstmParser()
            self.sequences = parser.parseSequences(subDir)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx]
    
    def numberOfFrames(self):
        return self.sequences[0].shape[0]

class lstmParser():
    def __init__(self, topDir='Pamonodaten'):
        self.topDir = topDir
    
    def parseSequences(self, imageDirectory):
        sequences = []
        samples =[]
        csvPath = os.path.join(self.topDir, imageDirectory, 'lstmGroundTruth.csv')
        with open(csvPath, newline='') as csvFile:
            reader = csv.reader(csvFile, delimiter=';')
            print('Loading samples...')
            for row in reader:
                array = np.asarray(row, dtype=np.float32)
                array[0:-1] = np.divide(array[0:-1], 65535, dtype=np.float32)
                samples.append(array)
            print('Samples successfully loaded.')
        return samples

    def __miniBatch(self, posSeq: list, negSeq: list):
        miniBatchSamples = []
        for i in range(len(posSeq)):
            random = np.random.randint(0, len(negSeq))
            miniBatchSamples.append(negSeq[random])
            negSeq.pop(random)
        miniBatchSamples += posSeq
        # print(len(miniBatchSamples))
        return miniBatchSamples

class lstmBoxParser():
    def __init__(self):
        pass

    def getBoxSize(self, csvPath):
        with open(csvPath) as csvFile:
            reader = csv.reader(csvFile, delimiter=';')
            for row in reader:
                return len(row)

    def getNumberOfTimesteps(self, csvPath):
        lineNr = 0
        with open(csvPath) as csvFile:
            reader = csv.reader(csvFile, delimiter=';')
            for row in reader:
                if row[0].startswith('Label'):
                    return lineNr
                else:
                    lineNr += 1

    def getMinMax(self, data):
        minSize = np.count_nonzero(data[0][0])
        for i in range(len(data)):
            nonZeroValues = np.count_nonzero(data[i][0]) 
            if nonZeroValues < minSize:
                minSize = nonZeroValues
        print(minSize)
        return minSize

    def parseBoxFeatures(self, subDir):
        csvPath = os.path.join('Pamonodaten', subDir, 'lstmGroundTruthv2.csv')
        boxSize = self.getBoxSize(csvPath)
        numberOfTImesteps = self.getNumberOfTimesteps(csvPath)
        data = []
        with open(csvPath) as csvFile:
            reader = csv.reader(csvFile, delimiter=';')
            lineNr = 0
            for row in reader:
                item = {}
                datapoint = np.asarray(row, dtype=np.float32)
                target = datapoint[-1]
                datapoint = np.divide(datapoint[0:-1], 65536, dtype=np.float32)
                std = np.std(datapoint[0:-1])
                mean = np.mean(datapoint[0:-1])
                for i in range(datapoint.shape[0]):
                    datapoint[i] = (datapoint[i] - mean ) / std
                item['sequence'] = datapoint
                item['target'] = target
                data.append(item)
        return data

    def parseAll(self):
        allData = []
        for dir in [dir for dir in os.listdir(os.path.join('Pamonodaten')) if os.path.isdir(os.path.join('Pamonodaten', dir))]:
            allData += self.parseBoxFeatures(dir)
        return allData

    def parseMultiple(self, subDirList):
        multipleData = []
        for dir in subDirList:
            multipleData += self.parseBoxFeatures(dir)
        return multipleData

if __name__ == "__main__":
    testDataset = LSTMDataset('PP_BG_eliminated', boxFeature=True)
    
    #print(testDataset[99], testDataset.timesteps, testDataset.boxSize)
    dataset = LSTMDataset('test',boxFeature=True)
    print(dataset[20])