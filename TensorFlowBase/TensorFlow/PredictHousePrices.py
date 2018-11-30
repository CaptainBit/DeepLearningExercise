from __future__ import absolute_import, division, print_function

import tensorflow as tf
from tensorflow import keras

import numpy as np

#Exercise link:
#https://www.tensorflow.org/tutorials/keras/basic_regression



def StartTutorial():

    bostonHousing = keras.datasets.bostonHousing
    (trainData, trainLabels), (testData, testLabels) = bostonHousing.load_data()


    #Shuffle the training set
    order = np.argsort(np.random.random(trainLabels.shape))
    trainData = trainData[order]


