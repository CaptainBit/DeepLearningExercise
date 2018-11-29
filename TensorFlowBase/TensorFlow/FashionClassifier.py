
#Link to tutorials
#https://www.tensorflow.org/tutorials/keras/basic_classification

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt




#Download/Import data set
def ImportDataSet():
    return keras.datasets.fashion_mnist

def StartTutorial():

    #labels name
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
                    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    #Import data set
    dataSet = ImportDataSet()

    #Dispatch
    (trainImages, trainLabels), (testImages, testLabels) = dataSet.load_data()


    #Show image in range of 0 to 255
    plt.figure()
    plt.imshow(trainImages[0])
    plt.colorbar()
    plt.grid(False)
    plt.title("Image between 0 and 255")
    plt.show()
    

    #Scale value between 0 to 1
    trainImages = trainImages/255.0
    testImages = testImages/255.0

    #reshow picture
    plt.figure()
    plt.imshow(trainImages[0])
    plt.colorbar()
    plt.grid(False)
    plt.title("Image between 0 and 1")
    plt.show()

    #Verif that every image is in good format
    plt.figure(figsize=(10,10))
    for i in range(25):
        plt.subplot(5,5,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(trainImages[i], cmap=plt.cm.binary)
        plt.xlabel(class_names[trainLabels[i]])

    plt.show()

    #linear stack of layers
    model = keras.Sequential([

        #put the image (2d array) to a 1d array
        keras.layers.Flatten(input_shape=(28,28)),

        #Dense of 128 nodes
        keras.layers.Dense(128, activation=tf.nn.relu), # output = activation(dot(input, kernel) + bias)
                                                        #where activation is the element-wise activation function passed as the activation argument, 
                                                        #kernel is a weights matrix created by the layer

                                                        #tf.nn.relu : (features,name=None)
                                                        #Computes rectified linear: max(features, 0)
                                                        #Activation func : defines the output of that node
                                                      
                                                        #A unit employing the rectifier is also called a rectified linear unit (ReLU)
                                                        #The derivative of ReLU:
                                                        #return x if x > 0 =>0 otherwise 
                                                        
        #Dense of 10 (10 possible result)
        keras.layers.Dense(10, activation=tf.nn.softmax)

        ])

    model.compile(optimizer=tf.train.AdamOptimizer(),  
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

    #training the model 
    model.fit(trainImages, trainLabels, epochs=5) #epochs = how many times the model will read the batch

    testLoss, testAcc = model.evaluate(testImages, testLabels)

    print('Test accuracy:', testAcc)


        









 
