import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
from Datasets.LSTMDataset import LSTMDataset
import numpy as np
import os
import csv


class LstmIntensity(nn.Module):
    def __init__(self, embeddingDim, hiddenDim, numberOfLayers):
        super(LstmIntensity, self).__init__()
        self.hiddenDim = hiddenDim
        self.lstm = nn.LSTM(embeddingDim, hiddenDim)
        self.hiddenToBinary = nn.Linear(hiddenDim, 1)  
        self.numberOfLayers = numberOfLayers
        self.hidden = self.init_hidden(1)

    def forward(self, intensitySequence):
        '''
        Input of shape (sequenceLength, batchSize, NumberOfFeatures)
        '''
        hiddenStates, self.hidden = self.lstm(intensitySequence, self.hidden)          # Outputs (All hidden states, most recent hidden state)
        mostRecentState = hiddenStates[-1]                      # (1 x HiddenSize)
        output = self.hiddenToBinary(mostRecentState)
        #output = F.sigmoid(output)           # (1 x OutputSize)
        return output

    def init_hidden(self, batch_size):
        return (autograd.Variable(torch.zeros(1, 1, self.hiddenDim)),
                autograd.Variable(torch.zeros(1, 1, self.hiddenDim)))

def getLSTMModel():
    model = LstmIntensity(1,2,1)
    return model

if __name__ == "__main__":       
    def trainLSTM(numberOfEpochs):
        
        trainDataset = LSTMDataset(['200nm_11Apr13_1'], boxFeature=True)
        testDataset = LSTMDataset(['PP_BG_eliminated'], boxFeature=True)
        model = LstmIntensity(1,2,1)

        optimizerAdam = optim.Adam(model.parameters(),lr = 0.005)
        #optimizerAdam = optim.SGD(model.parameters(), lr=0.01)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizerAdam, 'min',patience=5,verbose=True,cooldown=5,threshold=0.000001,factor=0.5)
       
        
        BCELoss =  nn.BCEWithLogitsLoss()
        #indices = torch.randperm(len(dataset)).tolist()
        print('Training Size:{}  Testing Size: {}'.format(len(trainDataset),len(testDataset)))
        #trainDataset = torch.utils.data.Subset(dataset, indices[:-300])                           #Abhängig von Größe des Datensatzes
        #testDataset = torch.utils.data.Subset(dataset, indices[-300:])

        trainDataloader = torch.utils.data.DataLoader(trainDataset, batch_size=1, num_workers=1, shuffle=True)
        testDataloader = torch.utils.data.DataLoader(testDataset, batch_size=1, num_workers=1, shuffle=False)

        evaluate(model, testDataloader,BCELoss)

        with open(os.path.join('Pamonodaten', 'lstmTraining.csv'),'w', newline='') as csvFile:
            writer = csv.writer(csvFile)
            writer.writerow(['trainLoss','testLoss'])
        
        bestLoss = 1
        for i in range(numberOfEpochs):
            print('Epoch [{}/{}]'.format(i+1, numberOfEpochs))
            trainLoss = trainOneEpoch(model, trainDataloader, BCELoss, optimizerAdam)
            print('Average loss: {}'.format(trainLoss))
            testLoss = evaluate(model, testDataloader,BCELoss)
            print('Learning Rate: {}'.format(optimizerAdam.param_groups[0]['lr']))

            with open(os.path.join('Pamonodaten', 'lstmTraining.csv'),'a', newline='') as csvFile:
                writer = csv.writer(csvFile)
                writer.writerow(['{}'.format(trainLoss),'{}'.format(testLoss)])
            
            scheduler.step(trainLoss)
            if testLoss < bestLoss:
                save(model, 'avgBrightness2')
            
        
    def trainOneEpoch(model: LstmIntensity, trainDataloader, lossFunction, optimizer):
        model.train()
        lossMetric = 0
        for index, item in enumerate(trainDataloader):
            model.hidden = model.init_hidden(1)
            model.zero_grad()
            # 3d input tensor (timeSteps x batches x featureSize)
            input = item['sequence'].unsqueeze(2).permute(1,0,2)
            target = item['target'].unsqueeze(1)
            output = model(input)
            loss = lossFunction(output, target)
            lossMetric += loss.item()
            loss.backward()
            optimizer.step()
        return lossMetric/index
        
                    

    def evaluate(model:LstmIntensity, testDataloader,lossFunction):
        with torch.no_grad():
            model.eval()

            acc = 0
            counter = 0
            lossMetric = 0
            for index, item in enumerate(testDataloader):
                model.hidden = model.init_hidden(1)
                input = item['sequence'].unsqueeze(2).permute(1,0,2)
                target = item['target'].unsqueeze(1)
                predLabel = 0
                #print(torch.sigmoid(model(input)[0]), target)
                output = model(input)
                if output >= 0:
                    predLabel = 1
                if predLabel ==  target:
                    acc += 1
                counter += 1
                lossMetric += lossFunction(output,target).item()
            print('Accuracy: {}'.format(acc / counter)) 
            return lossMetric / counter

    def save(model, saveFileName):
        torch.save(model.state_dict(), os.path.join('Results', saveFileName+'LSTM.pth'))


    trainLSTM(100)
#dataset = LSTMDataset('PP_BG_eliminated')
# 
# model = LstmIntensity(1,4,100)
# print(model(test), model(test).size())
