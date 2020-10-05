#-*- coding: utensorflow-8 -*-

#Run in server
#three-layer perceptron Neural Network

'''
input > 
weights 1 > convolution 1 > activation 1 > pooling 1 > 
weights 2 > convolution 2 > activation 2 > pooling 2 >
weights 3 > fully_connected_perceptron 3 > activation 3 >
output

compare output to intended output > cost function (cross entropy)
optimization function (optimizer) > minimize cost (AdamOptimizer....SGD, AdaGrad)

back propagation

feed forward + back propagation = 1 epoch
'''

import tflearn
from trlearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
import tflearn.datasets.mnist as mnist

X,Y,test_x,test_y = mnist.load_data(one_hot=True)


# one-hot -->
"""
0 = [1,0,0,0,0,0,0,0,0,0]
1 = [0,1,0,0,0,0,0,0,0,0]
2 = [0,0,1,0,0,0,0,0,0,0]
3 = [0,0,0,1,0,0,0,0,0,0]
4 = [0,0,0,0,1,0,0,0,0,0]
5 = [0,0,0,0,0,1,0,0,0,0]
6 = [0,0,0,0,0,0,1,0,0,0]
7 = [0,0,0,0,0,0,0,1,0,0]
8 = [0,0,0,0,0,0,0,0,1,0]
9 = [0,0,0,0,0,0,0,0,0,1]
"""

# pic size = 28x28 [height x width] = 784 pixels,
# convolution this time we don't flatten except for the fully connected layer
img_height = 28
img_width = 28
pixels = 784
n_classes = 10
batch_size = 128
color_channels = 1 #Grayscale

#window assumed square
pool_1_window_size = 2
pool_2_window_size = 2
compressed_img_height = img_height/pool_1_window_size_height/pool_2_window_size_height
compressed_img_width = img_width/pool_1_window_size_witdh/pool_2_window_size_witdh

#window assumed square
conv_1_window_size = 5
conv_2_window_size = 5

n_inputs_conv_layer_1 = 1
n_nodes_conv_layer_1 = 32
n_nodes_conv_layer_2 = 64
# input changes for fully connected
n_inputs_fc_layer_3 = compressed_img_width*compressed_img_height*n_nodes_conv_layer_2
n_nodes_fc_layer_3 = 1024

X = X.reshape([-1,img_width,img_height,color_channels])
test_x = test_x.reshape([-1,img_width,img_height,color_channels])

#### Model defining
def Convolutional_Model():
    convnet = input_data(shape=[None,img_width,img_height,color_channels], name='input')
    # Convolution Layer 1
    convnet = conv_2d(convnet,n_nodes_conv_layer_1,conv_1_window_size,strides=[1,1,1,1],activation='relu')
    convnet = max_pool_2d(convnet,pool_1_window_size,strides=[1,2,2,1])
    #Window size can be integer if square, or list [width, height]
    ###
    # Convolution Layer 2
    convnet = conv_2d(convnet,n_nodes_conv_layer_2,conv_2_window_size,activation='relu')
    convnet = max_pool_2d(convnet,pool_2_window_size,strides=[1,2,2,1])
    #Window size can be integer if square, or list [width, height]
    ###
    # Fully connected layer 3
    convnet = fully_connected(convnet, n_nodes_fc_layer_3, activation='relu')
    ###
    # Dropout technique
    dropout = False
    keep_rate = 0.8
    if dropout:
        convnet = dropout(convnet, keep_rate)
    ###
    # Output layer
    convnet = fully_connected(convnet, n_classes, activation='softmax')
    convnet = regression(
        convnet, 
        optimizer='adam', 
        learning_rate=0.01,
        loss='categorical_crossentropy',
        name='targets'
        )
    # Set the model to follow this network
    model = tflearn.DNN(convnet)
    return model

#################################
# If running for the first time #
#################################

model = Convolutional_Model()
model.fit(
    {'input': X},
    {'targets': Y},
    n_epoch=10,
    validation_set=({'input':test_x},{'targets':test_y}),
    show_metric=True,
    batch_size=batch_size,
    shuffle=False,
    snapshot_step=500,
    run_id='mnist'
    )
#Only saves weights, we still have to run Convolutional_Model() to load again
model.save('tflearncnn.model') 
    
####################
# If loading model #
####################

model = Convolutional_Model()
model.load('tflearncnn.model')

print(model.predict([test_x[1]]))
