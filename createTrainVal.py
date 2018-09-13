#！/usr/bin/env python
# _*_ coding:utf-8 _*_

import os
import json
import shutil

os.chdir('../')

currentDir = os.getcwd()


oriValImageDataDir = currentDir + '/data/AgriculturalDisease_validationset/images'
oriValLabelDataDir = currentDir + '/data/AgriculturalDisease_validationset/AgriculturalDisease_validation_annotations.json'

tarValImageDataDir = currentDir + '/data/val'

oriTrainImageDataDir = currentDir + '/data/AgriculturalDisease_trainingset/images'
oriTrainLabelDataDir = currentDir + '/data/AgriculturalDisease_trainingset/AgriculturalDisease_train_annotations.json'
tarTrainImageDataDir = currentDir + '/data/train'

def createClassData( is_train=True ):
    def readLabelFromJson(dataDir):
        with open(dataDir, 'r') as f:
            resultList = json.load(f)
        return resultList

    if is_train:
        dataJsonDir = oriTrainLabelDataDir
        oriImageDataDir = oriTrainImageDataDir
        tarImageDataDir = tarTrainImageDataDir

    else:
        dataJsonDir = oriValLabelDataDir
        oriImageDataDir = oriValImageDataDir
        tarImageDataDir = tarValImageDataDir


    ##### is a list, element is a dict {'image_id': xxx.jpg, 'disease_class':1}
    LabelList = readLabelFromJson( dataJsonDir )
    if is_train:
        print('trian data number is: {}'.format(len(LabelList)))
    else:
        print('val data number is: {}'.format(len(LabelList)))

    for ii in range(len(LabelList)):
        tempValLabel = LabelList[ii]
        tempClassDir = tarImageDataDir + '/' + str( tempValLabel['disease_class'] )
        if not os.path.exists( tempClassDir ):
            os.makedirs( tempClassDir )
        tempOriImageName = oriImageDataDir + '/' + str( tempValLabel['image_id'] )
        tempTarImageName = tempClassDir + '/'
        shutil.copy(tempOriImageName, tempTarImageName)
        if (ii + 1) % 1000 == 0:
            print('{} is done'.format(ii + 1))

createClassData(is_train=True)
#createClassData(is_train=False)