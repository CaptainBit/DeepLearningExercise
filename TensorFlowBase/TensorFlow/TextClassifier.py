
#Link to tutorials
#https://www.tensorflow.org/tutorials/keras/basic_text_classification

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt


import matplotlib.pyplot as plt

#Good link activation func:
#https://towardsdatascience.com/activation-functions-neural-networks-1cbd9f8d91d6

#Good link Cross-Entropy:
#https://ml-cheatsheet.readthedocs.io/en/latest/loss_functions.html

#Binary classification:
    #classifying the elements into two groups

def StartTutorial():

    imdb = keras.datasets.imdb

    #Import the data
    (trainData,trainLabels), (testData, testLabels) = imdb.load_data(num_words = 10000)

    #Example of what a review of a movie can look like
    wordIndex = imdb.get_word_index()

    wordIndex = {k:(v+3) for k,v in  wordIndex.items()} 
    wordIndex["<PAD>"] = 0
    wordIndex["<START>"] = 1
    wordIndex["<UNK>"] = 2  # unknown
    wordIndex["<UNUSED>"] = 3


    reverseWordIndex = dict([(value, key) for (key, value) in  wordIndex.items()])
     
    def decodeReview(text):
        return ' '.join([reverseWordIndex.get(i, '?') for i in text]) 
    

    #Example of review
    print(decodeReview(trainData[0]))

    #Movie reviews must be the same length
    trainData = keras.preprocessing.sequence.pad_sequences(trainData,
                                                           value=wordIndex["<PAD>"],
                                                           padding='post',
                                                           maxlen=256)

    testData = keras.preprocessing.sequence.pad_sequences(testData,
                                                          value=wordIndex["<PAD>"],
                                                          padding='post',
                                                          maxlen=256)
    #Check if the length


    #print("lenght  of 0: " + str(len(trainData[0])))
    #print("lenght of 1: " + str(len(trainData[1])))
      


    vocabSize = 10000 #vocabulary count for the movie review (10,000 most frequently occurring words in the training data)

    model = keras.Sequential()

    model.add(keras.layers.Embedding(vocabSize, 16)) #looks up the embedding vectors for each word-index. (learned as the model trains)
                                                     #

##################################################################################################################################
                                                                                                                                    #
    model.add(keras.layers.GlobalAveragePooling1D()) #return fixed-length vector by averaging over sequence dimension               ##
                                                     #Allow model to handle input of variable length. in the simplest way possible  ## Hidden layers 
                                                                                                                                    ##
    model.add(keras.layers.Dense(16, activation=tf.nn.leaky_relu))                                                                  #
                                                      #Counter the 'dying relu'
##################################################################################################################################
    
    model.add(keras.layers.Dense(1, activation=tf.nn.sigmoid))#use it to predict probability between 0 to 1 (sigmoid return value between 0 and 1 only)

    #Show model 
    model.summary()

    #Cross-Entropy loss func : loss increases as the predicted probability diverges from the actual label. (exponential => As the predicted probability decreases, the log loss increases rapidly )
    model.compile(optimizer=tf.train.AdamOptimizer(),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    #Create a validation set (first 10 000)
    xVal = trainData[:10000]
    #starting at 10 000
    partialXTrain = trainData[10000:]

    yVal = trainLabels[:10000]
    partialYTrain = trainLabels[10000:]

    #train the ai
    history = model.fit(
        partialXTrain,
        partialYTrain,
        epochs=40,
        batch_size=512,
        validation_data=(xVal,yVal),
        verbose=1)
    #validation
    results = model.evaluate(testData, testLabels)

    print(results)
    
    historyDict = history.history
    historyDict.keys()


    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(1, len(acc) + 1)

    # "bo" is for "blue dot"
    plt.plot(epochs, loss, 'bo', label='Training loss')
    # b is for "solid blue line"
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.show()


    plt.clf()   # clear figure
    acc_values = historyDict['acc']
    val_acc_values = historyDict['val_acc']

    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.show()



 

             




