
#Link to tutorials
#https://www.tensorflow.org/tutorials/keras/basic_text_classification

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt






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
    print("lenght  of 0: " + str(len(trainData[0])))
    print("lenght of 1: " + str(len(trainData[1])))
      


    vocabSize = 10000 #vocabulary count for the movie review (10,000 most frequently occurring words in the training data)

    model = keras.Sequential()

    model.add(keras.layers.Embedding(vocabSize, 16)) #looks up the embedding vectors for each word-index. (learned as the model trains)
    model.add(keras.layers.GlobalAveragePooling1d())
    








    

 

             




