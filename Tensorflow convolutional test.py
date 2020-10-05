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

"""
Tensorflow prepares a "calculation graph",
which has all the calculations laid out, but doesn't run them yet,
a session has to be open and run and then closed to actually calculate stuff
"""

import tensorflow
from tensorflow.examples.tutorials.mnist import input_data
from __future__ import print_function

# mnist --> handwritten digit recognition
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

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

# When pooling, the feature size is compressed 
# in feature numbers from total pixels to the following
pool_1_window_size_height = 2
pool_1_window_size_witdh = 2
pool_2_window_size_height = 2
pool_2_window_size_witdh = 2
compressed_img_height = img_height/pool_1_window_size_height/pool_2_window_size_height
compressed_img_width = img_width/pool_1_window_size_witdh/pool_2_window_size_witdh

conv_1_window_size_width = 5
conv_1_window_size_height = 5
conv_2_window_size_width = 5
conv_2_window_size_height = 5

# tensorflow uses placeholders before running
x = tensorflow.placeholder('float',[None, pixels])
y = tensorflow.placeholder('float')

#Dropout technique
keep_rate = 0.8


### Model variables storage is dictionaries:
# per layer y = wx + b (w: weights, b: biases)
conv_layer_1 = dict()
conv_layer_2 = dict()
fc_layer_3 = dict()
output_layer = dict()

n_inputs_conv_layer_1 = 1
n_nodes_conv_layer_1 = 32
n_nodes_conv_layer_2 = 64
# input changes for fully connected
n_inputs_fc_layer_3 = compressed_img_width*compressed_img_height*n_nodes_conv_layer_2
n_nodes_fc_layer_3 = 1024

def conv_2d(x,W):
    # x: input
    # W: weights
    # strides = moving convlolution window (like small chunks of an image)
    # [1,1,1,1] is [1 image, 1 pix width, 1 pix height, 1 color channel]
    # padding is if the window is haniging empty over the image, same fills empty space with last values
    return tensorflow.nn.conv2d(x, W, strides=[1,1,1,1],padding='SAME')

def max_pool_2d(x):
    # x: input (after convoluted)
    # ksize: size of window
    # strides: moving of pooling window
    # [1,2,2,1] is [1 image, 2 pix width, 2 pix height, 1 color channel]
    # padding is if the window is haniging empty over the image, same fills empty space with last values
    return tensorflow.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')


def Define_Model():
    conv_layer_1 = {
    'weights': tensorflow.Variable(tensorflow.random_normal([conv_1_window_size_width,conv_1_window_size_height,n_inputs_conv_layer_1,n_nodes_conv_layer_1])),
    'biases':tensorflow.Variable(tensorflow.random_normal([n_nodes_conv_layer_1]))
    }
    conv_layer_2 = {
    'weights': tensorflow.Variable(tensorflow.random_normal([conv_2_window_size_width,conv_2_window_size_height,n_nodes_conv_layer_1,n_nodes_conv_layer_2])),
    'biases':tensorflow.Variable(tensorflow.random_normal([n_nodes_conv_layer_2]))
    }
    fc_layer_3 = {
    'weights': tensorflow.Variable(tensorflow.random_normal([n_inputs_fc_layer_3,n_nodes_fc_layer_3])),
    'biases':tensorflow.Variable(tensorflow.random_normal([n_nodes_fc_layer_3]))
    }
    output_layer = {
    'weights': tensorflow.Variable(tensorflow.random_normal([n_nodes_fc_layer_3, n_classes])),
    'biases':tensorflow.Variable(tensorflow.random_normal([n_classes]))
    }

def neural_network_forward(x_input_data, dropout=False):
    # input reshape
    x_input_data = tensorflow.reshape(x,shape=[-1, img_width, img_height, color_channels]) #-1 is batch_size automatically

    # Convolution layer 1
    conv1 = tensorflow.add(conv_2d(x_input_data, conv_layer_1['weights']),conv_layer_1['biases'])
    # Activation function 1
    conv1 = tensorflow.nn.relu(conv1)
    # Pooling layer 1
    pool1 = max_pool_2d(conv1)


    # Convolution layer 2
    conv2 = tensorflow.add(conv_2d(pool1, conv_layer_2['weights']),conv_layer_2['biases'])
    # Activation function 2
    conv2 = tensorflow.nn.relu(conv2)
    # Pooling layer 2
    pool2 = max_pool_2d(conv2)

    # input reshape
    reshaped_input = tensorflow.reshape(pool2,[-1,n_inputs_fc_layer_3])
    # Fully connected layer 3
    fc3 = tensorflow.add(tensorflow.matmul(reshaped_input,fc_layer_3['weights']), fc_layer_3['biases'])
    # Activation function
    fc3 = tensorflow.nn.relu(fc3)

    #Dropout technique
    if dropout:
        fc3 = tensorflow.nn.dropout(fc3,keep_rate)

    # Output layer
    output = tensorflow.add(tensorflow.matmul(fc3,output_layer['weights']), output_layer['biases'])

    return output

# Tensorflow saver object allows to save model, but have to preload
saver = tensorflow.train.Saver()
# saver.save(sess, "model.ckpt") #Checkpoing saving of model
# saver.restore(sess, "model.ckpt") #Checkpoint loading of model

def train_neural_network(x, hm_epochs=10,checkpoint_path=None):
    # Initial Forward Propagation
    prediction = neural_network_forward(x)
    # Calculate "error" as cost, average of cross entropies
    cost = tensorflow.reduce_mean(tensorflow.nn.softmax_cross_entropy_with_logits(logits=prediction,labels=y))
    # Optimizer that reduces cost, "minimize error"
    optimizer = tensorflow.train.AdamOptimizer().minimize(cost) 
    # tensorflow prepares already knows how to minimize
    
    # 1 epoch = feed forward + back propagation
    # hm_epochs = 10
    #
    #Session
    #
    with tensorflow.Session() as sess:
        sess.run(tensorflow.global_variables_initializer())
        if checkpoint_path:
            saver.restore(sess, checkpoint_path)
        for epoch in xrange(hm_epochs):
            epoch += 1
            epoch_loss = 0
            for _ in range(int(mnist.train.num_examples/batch_size)):
                # batch taking is included in mnist, 
                # but in my own data for other projects i have to build it 
                epoch_x,epoch_y = mnist.train.next_batch(batch_size)
                _,c = sess.run([optimizer,cost],feed_dict = {x: epoch_x, y: epoch_y})
                epoch_loss += c
            saver.save(sess, "model.ckpt") #Checkpoing saving of model
            print("Epoch",epoch,"completed out of",hm_epochs,"loss:",epoch_loss)
        # tensorflow.argmax returns index of maximum values, since this is one-hot, 
        # to tell us which number is predicted vs which number is the y label
        # tensorflow.equal is just tensorflow equivalent of x == y = True or False
        correct = tensorflow.equal(tensorflow.argmax(prediction,axis=1),tensorflow.argmax(y,axis=1)) 
        #axis= 1 is vertical across data, meaning, maximum of each row across all rows. size is size of data samples
        #axis= 0 would be horizontal, size is number of features, not samples.
        ####
        # tensorflow.reduce_mean averages stuff
        # tensorflow.cast changes variable type from tensor to float
        accuracy = tensorflow.reduce_mean(tensorflow.cast(correct,'float'))
        print("Accuracy:", accuracy.eval({x:mnist.test.images, y:mnist.test.labels}))
    return checkpoint_path

def neural_network_prediction(input_data, checkpoint_path):
    # input data must be in format of [[],[],[]]. 
    # If it's one sample it's [[]]
    # if it was one sample in shape [], axis should be 0 instead of 1.
    prediction = neural_network_forward(input_data)
    with tensorflow.Session() as sess:
        sess.run(tensorflow.global_variables_initializer())
        saver.restore(sess, checkpoint_path)
        result = sess.run(tensorflow.argmax(prediction,axis=1))
    return result

Define_Model()
checkpoint = train_neural_network(x, hm_epochs=10)
test_x = [mnist.test.images[0]]
test_y = mnist.test.labels[0]
predicted = neural_network_prediction(test_x, checkpoint_path)
print(predicted)
print(test_y)


################################################
################################################
################################################
################################################

def cnn_model_fn(features, labels, mode):
    # features in shape of {'x':x}
    """Model function for CNN."""
    # Input Layer
    input_layer = tensorflow.reshape(features["x"], [-1, 28, 28, 1])

    # Convolutional Layer 1
    conv1 = tensorflow.layers.conv2d(
        inputs=input_layer,
        filters=32,
        kernel_size=[5, 5],
        padding="same",
        activation=tensorflow.nn.relu)
  
    # Pooling Layer 1
    pool1 = tensorflow.layers.max_pooling2d(
        inputs=conv1, 
        pool_size=[2, 2], 
        strides=2
        )
  
    # Convolutional Layer 2
    conv2 = tensorflow.layers.conv2d(
        inputs=pool1,
        filters=64,
        kernel_size=[5, 5],
        padding="same",
        activation=tensorflow.nn.relu)
    # Pooling Layer 2
    pool2 = tensorflow.layers.max_pooling2d(
        inputs=conv2, 
        pool_size=[2, 2], 
        strides=2
        )
  
    # Fully connected Layer 3
    # Reshaping input
    pool2_flat = tensorflow.reshape(pool2, [-1, 7 * 7 * 64])
    #
    fc3 = tensorflow.layers.dense(inputs=pool2_flat, units=1024, activation=tensorflow.nn.relu)
    dropout = False
    if dropout:
        fc3 = tensorflow.layers.dropout(
            inputs=dense, rate=0.4, training=mode == tensorflow.estimator.ModeKeys.TRAIN)
      
    # Outuput Layer
    output = tensorflow.layers.dense(inputs=fc3, units=10)

    predictions = {
        # Generate predictions (for PREDICT and EVAL mode)
        "classes": tensorflow.argmax(input=output, axis=1),
        # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
        # `logging_hook`.
        "probabilities": tensorflow.nn.softmax(output, name="softmax_tensor")
    }
  
    if mode == tensorflow.estimator.ModeKeys.PREDICT:
      return tensorflow.estimator.EstimatorSpec(mode=mode, predictions=predictions)
  
    # Calculate Loss (for both TRAIN and EVAL modes)
    onehot_labels = tensorflow.one_hot(indices=tensorflow.cast(labels, tensorflow.int32), depth=10)
    loss = tensorflow.losses.softmax_cross_entropy(
        onehot_labels=onehot_labels, logits=output)
  
    # Configure the Training Op (for TRAIN mode)
    if mode == tensorflow.estimator.ModeKeys.TRAIN:
      optimizer = tensorflow.train.GradientDescentOptimizer(learning_rate=0.001)
      train_op = optimizer.minimize(
          loss=loss,
          global_step=tensorflow.train.get_global_step())
      return tensorflow.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)
  
    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
        "accuracy": tensorflow.metrics.accuracy(
            labels=labels, predictions=predictions["classes"])}
    return tensorflow.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

    
