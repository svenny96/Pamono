import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, RPNHead
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from PytorchCoco.engine import train_one_epoch, evaluate
from PytorchCoco import utils
from PytorchCoco import transforms as T
import torch
from Datasets.RCNNDataset import RCNNDataset
from Datasets.PennFudanDataset import PennFudanDataset
import numpy
from torchvision import transforms
from PIL import Image, ImageDraw, ImageOps
import os
import cv2
import miscUtils as misc
from functools import reduce
import csv


def getCustomModel():
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    '''
    Aug2016 : 
    Anchors: 140, 150, 160, 170, 180
    Aspect Ratios: 0.8, 0.9, 1, 1.1, 1.2

    Aerosol:
    Anchors : 10, 12, 14, 16
    Ratios : 3, 3.5, 4, 4.5
    '''
    
    anchor_generator = AnchorGenerator(sizes=tuple([(30, 40, 50, 60, 70) for _ in range(5)] ), aspect_ratios=tuple([(0.25, 0.5, 1, 2, 4)for _ in range(5)])) #75 ?
    rpn_head = RPNHead(256, anchor_generator.num_anchors_per_location()[0])
    print(anchor_generator.num_anchors_per_location())
    model.rpn.anchor_generator = anchor_generator
    model.rpn.head = rpn_head
    
    
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)
    return model

def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0))
    return T.Compose(transforms)

def save(model, num_epochs, dataLoader, device, modelName = "\epochmodel"):
    torch.save(model.state_dict(), os.path.join("Results", modelName+'.pth'))
    model.eval()
    
def pamonoEvaluate(model, dataLoader):
    iouSum = []
    predBoxesNr = 0
    sum_loss = 0
    loss_classifier = 0
    loss_box_reg = 0
    loss_objectness = 0
    loss_rpn_box_reg = 0
    numel = 0
    for index, item in enumerate(dataLoader):
        with torch.no_grad():
            model.train()
            tensorList = []
            tensorTargetList = []
            tensorList.append(item[0][0].cuda())
            targets = item[1][0]
            for key in targets:
                targets[key] = targets[key].cuda()
            tensorTargetList.append(targets)
            loss = model(tensorList,tensorTargetList)
            loss_classifier += loss['loss_classifier'].item()
            loss_box_reg += loss['loss_box_reg'].item()
            loss_objectness += loss['loss_objectness'].item()
            loss_rpn_box_reg += loss['loss_rpn_box_reg'].item()
            sum_loss += loss['loss_classifier'].item()+loss['loss_box_reg'].item()+loss['loss_objectness'].item()+loss['loss_rpn_box_reg'].item()
            
            numel += 1
    return sum_loss / numel
    #'loss_classifier','loss_box_reg', 'loss_objectness', 'loss_rpn_box_reg'
            
    # tensorImage = numpy.array(numpy.divide(croppedImage, 65535), dtype=numpy.float32) 
    # tensorImage = torch.from_numpy(tensorImage)
    # tensorList.append(tensorImage.unsqueeze(0).cuda())
    # output = self.RCNNModel(tensorList)

if __name__ == "__main__":
    def main(saveModel=False, num_epochs=1):
        device =  torch.device('cuda')

        dataset = RCNNDataset("Pamonodaten", get_transform(train=True),subDir=['200nm_11Apr13_1'])
        testDataset = RCNNDataset("Pamonodaten", get_transform(train=False),subDir=['PP_BG_eliminated'])

        #dataset = PennFudanDataset("PennFudanPed", get_transform(train=True))
        #testDataset = PennFudanDataset("PennFudanPed", get_transform(train=False))

        
        indices = torch.randperm(len(testDataset)).tolist()
        #dataset = torch.utils.data.Subset(dataset, indices[:-100])                           #Abhängig von Größe des Datensatzes
        testDataset = torch.utils.data.Subset(testDataset, indices[-300:])
        print('Training Size: {}  Testing Size: {}'.format(len(dataset),len(testDataset)))
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True, num_workers=1,collate_fn=utils.collate_fn) 
        dataloaderTest = torch.utils.data.DataLoader(testDataset, batch_size=1, shuffle=False, num_workers=1,collate_fn=utils.collate_fn)

        model = getCustomModel()
        model.to(device)
        
        """ with open(os.path.join('Pamonodaten', 'fasterRcnnStats.csv'),'w', newline='') as csvFile:
            writer = csv.writer(csvFile)
            for i in range(fpr.shape[0]):
                 writer.writerow(['{testLoss}','{valLoss}'])
        """

        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
        # and a learning rate scheduler
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
        print(device)

        with open(os.path.join('Pamonodaten', 'fasterRCNNtrain.csv'),'w', newline='') as csvFile:
            writer = csv.writer(csvFile)
            writer.writerow(['trainLoss'])
        
        with open(os.path.join('Pamonodaten', 'fasterRCNNtest.csv'),'w', newline='') as csvFile:
            writer = csv.writer(csvFile)
            writer.writerow(['testLoss'])
        
        testLoss = pamonoEvaluate(model,dataloaderTest)
        with open(os.path.join('Pamonodaten', 'fasterRCNNtest.csv'),'a', newline='') as csvFile:
            writer = csv.writer(csvFile)
            writer.writerow(['{}'.format(testLoss)])
        #pamonoEvaluate(model,dataloaderTest)
        for epoch in range(num_epochs):
            train_one_epoch(model, optimizer, dataloader, device, epoch, print_freq=10)
            # update the learning rate
            lr_scheduler.step()
            # evaluate on the test dataset
            evaluate(model, dataloaderTest, device=device)
            testLoss = pamonoEvaluate(model, dataloaderTest)

            with open(os.path.join('Pamonodaten', 'fasterRCNNtest.csv'),'a', newline='') as csvFile:
                writer = csv.writer(csvFile)
                writer.writerow(['{}'.format(testLoss)])

            save(model,num_epochs, dataloaderTest, device, 'final')
            print('saved pth')
        
    epochs = 10
    main(saveModel=True, num_epochs=epochs)
    #demo.loadModel(dataLoaderTest, 'Results\epoch{}model.pth'.format(epochs), 'Results\epoch{}model'.format(epochs))