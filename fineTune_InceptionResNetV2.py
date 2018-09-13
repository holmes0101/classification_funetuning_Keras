#！/usr/bin/env python
# _*_ coding:utf-8 _*_

#！/usr/bin/env python
# _*_ coding:utf-8 _*_

from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, Dropout
from keras import backend as K
from keras.layers import Input
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
import os
from keras.callbacks import ModelCheckpoint

os.chdir('../')
currentDir = os.getcwd()

InResNetV2Dir = '/home/holmes/keras_model/inception_resnet_v2/inception_resnet_v2_weights_tf_dim_ordering_tf_kernels_notop.h5'
trainDataDir = currentDir + '/data/train'
valDataDir = currentDir + '/data/val'


base_model = InceptionResNetV2(weights = InResNetV2Dir, include_top = False, pooling = 'avg')
base_modelOutput = base_model.output
x = Dense(1024, activation = 'relu')(base_modelOutput)
predictions = Dense(61, activation = 'softmax')( x )
model = Model(inputs = base_model.input, outputs = predictions)
'''
for layer in base_model.layers:
    layer.trainable = False
'''
model.compile( loss = 'binary_crossentropy', optimizer = SGD(lr = 1e-3, momentum = 0.9), metrics = ['accuracy'] )

train_dataGen = ImageDataGenerator( rescale = 1. / 255, shear_range = 0.2, zoom_range = 0.2, horizontal_flip = True )
val_dataGen = ImageDataGenerator( rescale = 1. / 255 )

modelDir = currentDir + '/data/model/InceptionResNetV2/InceptionResNetV2-ep{epoch:03d}-acc-{acc:.5f}-loss{loss:.5f}-val_acc:{val_acc:.5f}-val_loss{val_loss:.5f}.h5'
if not os.path.exists(currentDir + '/data/model/InceptionResNetV2'):
    os.makedirs(currentDir + '/data/model/InceptionResNetV2')

classesNumber = 61
classes = []
for imageName in range(0, classesNumber):
    classes.append( str(imageName) )

checkpoint = ModelCheckpoint( modelDir, monitor='val_loss', verbose=1, save_best_only= True, save_weights_only=False, mode='min', period=1 )

trainGenerator =  train_dataGen.flow_from_directory( trainDataDir, target_size = (299, 299), batch_size = 4, classes = classes )
valGenerator = val_dataGen.flow_from_directory( valDataDir, target_size = (299, 299), batch_size = 2, shuffle = False, classes = classes  )
model.fit_generator( trainGenerator, steps_per_epoch = 8200, epochs = 100, verbose = 1,
                     validation_data = valGenerator, validation_steps = 2491, callbacks=[checkpoint])
modelDir = currentDir + '/data/model/InceptionResNetV2/InceptionResNetV2-finetune_1024_last_epoch.h5'
model.save( modelDir )