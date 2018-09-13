#！/usr/bin/env python
# _*_ coding:utf-8 _*_

from keras.applications.xception import Xception
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, Dropout
from keras import backend as K
from keras.layers import Input
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
import os
import numpy as np
from keras.models import load_model
import pandas as pd
import json
import time

os.chdir('../')
currentDir = os.getcwd()
from PIL import Image
modelDir = currentDir + '/data/model/Xception-ep006-loss0.03856-val_loss0.03204.h5'
model = load_model( modelDir )
testDataDir = currentDir + '/data/AgriculturalDisease_testA/images'


curTime = time.strftime('%Y-%m-%d-%H-%M', time.localtime(time.time()))
imageNameList = []
predScoreList = []
predLabelList = []
resultList = []
counter = 0
for imageName in os.listdir(testDataDir):
    counter += 1
    testImageName = testDataDir + '/' + str(imageName)
    testImage = image.load_img( testImageName, target_size = (299, 299) )
    testImage = image.img_to_array( testImage )
    #print(testImage)
    #break
    testImage = np.double( testImage ) / 255.0
    testImage = np.expand_dims( testImage, axis = 0 )
    pred = model.predict( testImage )
    imageNameList.append( str(imageName) )
    if counter % 100 ==0:
        print( "{} is done".format( counter ) )
    tempPred = pred[0]
    predScoreList.append( tempPred )
    maxIndex = np.argmax( tempPred )
    predLabelList.append( maxIndex )
    tempDict = {}
    tempDict['image_id'] = str(imageName)
    tempDict['disease_class'] = int(maxIndex)
    resultList.append( tempDict )
    #print(tempPred)
    #print( tempPred[maxIndex] )
    #print( maxIndex )
    #ScoreList.append( pred[0][1] )
    #raise('test')
    #break

if not os.path.exists(currentDir + '/Xception-result'):
    os.makedirs(currentDir + '/Xception-result')

resultCSVDir = currentDir + '/Xception-result/' + str(curTime) + 'Xception-finetune1024' + '.csv'
data = pd.DataFrame( {'image_id': imageNameList, 'pred_score': predScoreList} )
data.to_csv( resultCSVDir, index = False )

resultJsonDir = currentDir + '/Xception-result/' + str(curTime) + 'Xception_finetune1024-' + '.json'

with open(resultJsonDir, 'w') as f:
    json.dump(resultList, f)
