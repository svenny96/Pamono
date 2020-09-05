import xml.dom.minidom
import xml.etree.ElementTree as ET
import os
import sys
from pathlib import Path
import csv

def getImages():
    pass

def parsePolygon(xmlPolygon):
    row = []
    points = xmlPolygon.split(';')
    for point in points:
        coords = point.split(',')
        row.append(coords[0])
        row.append(coords[1])
    while len(row) < 64:
        row.append('NaN')
    return row

subDir = Path(sys.argv[1])
for file in subDir.iterdir():
    if file.suffix == '.xml':
        xmlFilePath = file

csvPath = subDir.joinpath('VirusDetectionCLPolygonFormFactors_positives.csv')

tree = ET.parse(xmlFilePath)
root = tree.getroot()
print(root.tag)
imageDicts = []
for image in root.iter('image'):
    imageDicts.append(image)

headerRow = ['frameNumber', 'fileName']
for i in range(64):
    headerRow.append('x{}'.format(i))
    headerRow.append('y{}'.format(i))

with open(csvPath, 'w', newline='') as csvFile:
    writer = csv.writer(csvFile, delimiter=';')
    writer.writerow(headerRow)
    for image in root.iter('image'):
        for polygon in image.iter('polygon'):
            writer.writerow([image.get('id'), image.get('name')] + parsePolygon(polygon.get('points')))