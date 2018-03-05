#-*- coding: utensorflow-8 -*-

#Run in server
#three-layer perceptron Neural Network

'''
input >
weights 1 > hidden layer 1 > activation function >
weights 2 > hidden layer 2 > activation function > 
weights 3 > 
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

n_nodes_hidden_layer_1 = 500
n_nodes_hidden_layer_2 = 500
n_nodes_hidden_layer_3 = 500

n_classes = 10
batch_size = 128

# pic size = 28x28 [height x width] = 784 pixels,
# but we can flatten the matrix to one row
features = 784
# which is size of the input data one sample

x = tensorflow.placeholder('float',[None, features])
y = tensorflow.placeholder('float')

### Model variables storage is dictionaries:
# per layer y = wx + b (w: weights, b: biases)
hidden_layer_1 = dict()
hidden_layer_2 = dict()
hidden_layer_3 = dict()
output_layer = dict()

def Define_Model():
    hidden_layer_1 = {
    'weights': tensorflow.Variable(tensorflow.random_normal([features, n_nodes_hidden_layer_1])),
    'biases':tensorflow.Variable(tensorflow.random_normal([n_nodes_hidden_layer_1]))
    }
    hidden_layer_2 = {
    'weights': tensorflow.Variable(tensorflow.random_normal([n_nodes_hidden_layer_1, n_nodes_hidden_layer_2])),
    'biases':tensorflow.Variable(tensorflow.random_normal([n_nodes_hidden_layer_2]))
    }
    hidden_layer_3 = {
    'weights': tensorflow.Variable(tensorflow.random_normal([n_nodes_hidden_layer_2, n_nodes_hidden_layer_3])),
    'biases':tensorflow.Variable(tensorflow.random_normal([n_nodes_hidden_layer_3]))
    }
    output_layer = {
    'weights': tensorflow.Variable(tensorflow.random_normal([n_nodes_hidden_layer_3, n_classes])),
    'biases':tensorflow.Variable(tensorflow.random_normal([n_classes]))
    }

def neural_network_forward(x_input_data):
    # Forward propagation layer 1
    l1 = tensorflow.add(tensorflow.matmul(x_input_data,hidden_layer_1['weights']), hidden_layer_1['biases'])
    # Activation function
    l1 = tensorflow.nn.relu(l1)

    # Forward propagation layer 2
    l2 = tensorflow.add(tensorflow.matmul(l1,hidden_layer_2['weights']), hidden_layer_2['biases'])
    # Activation function
    l2 = tensorflow.nn.relu(l2)

    # Forward propagation layer 3
    l3 = tensorflow.add(tensorflow.matmul(l2,hidden_layer_3['weights']), hidden_layer_3['biases'])
    # Activation function
    l3 = tensorflow.nn.relu(l3)

    # Output layer
    output = tensorflow.add(tensorflow.matmul(l3,output_layer['weights']), output_layer['biases'])

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




