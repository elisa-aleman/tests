#-*- coding: utf-8 -*-

#Run in server
#three-layer perceptron Neural Network

import numpy as np
import chainer
from chainer import cuda, Function, gradient_check, report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions

########################
# 1. Prepare a dataset #
########################

# test dataset handrwitten digit recognition
from chainer.datasets import mnist

train, test = mnist.get_mnist(withlabel=True, ndim=1)

# Display an example from the MNIST dataset.
# `x` contains the input image array and `t` contains that target class
# label as an integer.
from __future__ import print_function
import matplotlib.pyplot as plt
x, t = train[0]
# type(x)
# <type 'numpy.ndarray'>
# type(t)
# <type 'numpy.int32'>
plt.imshow(x.reshape(28, 28), cmap='gray')
plt.savefig('5.png')
print('label:', t)

################################
# 2. Create a dataset iterator #
################################

# Iterator class that retrieves a set of data and labels
# from the given dataset to easily make a mini-batch. 
from chainer import iterators

batchsize = 128
train_iter = iterators.SerialIterator(train, batchsize)
test_iter = iterators.SerialIterator(test, batchsize, repeat=False, shuffle=False)

#######################
# 3. Define a network #
#######################

n_nodes_hidden_layer_1 = 500
n_nodes_hidden_layer_2 = 500
n_nodes_hidden_layer_3 = 500
n_classes = 10
features = 784
batch_size = 128

class MyNetwork(Chain):
    def __init__(
        self, 
        n_nodes_hidden_layer_1=500,
        n_nodes_hidden_layer_2 = 500,
        n_nodes_hidden_layer_3 = 500,
        n_classes = 10,
        
        ):
        super(MyNetwork, self).__init__()
        with self.init_scope():
            # Fully connected layer 1
            self.l1 = L.Linear(None, n_nodes_hidden_layer_1) # first linear perceptron, y = weights*x + bias
            #
            # input size=None defaults to features = 784 when forward propagation
            # The weight matrix W is initialized with i.i.d. Gaussian samples, 
            # each of which has zero mean and deviation 1/in_size‾‾‾‾‾‾‾‾√. 
            # The bias vector b is of size out_size. Each element is initialized with the bias value. 
            # If nobias argument is set to True, then this link does not hold a bias vector.
            #
            # Fully connected layer 2
            self.l2 = L.Linear(n_nodes_hidden_layer_1, n_nodes_hidden_layer_2) # second linear perceptron
            # Fully connected layer 3
            self.l3 = L.Linear(n_nodes_hidden_layer_2, n_nodes_hidden_layer_3) #third linear perceptron
            # Output fully connected layer
            self.out = L.Linear(n_nodes_hidden_layer_3, n_classes) #output linear perceptron (output)
            #L.Linear object has __call__(x) function to forward propagate the x by weights and bias

    def __call__(self, x): # Forward propagation (x)
        # Rectified Linear Unit function (activation function)
        hidden_layer_1 = F.relu(self.l1(x))
        hidden_layer_2 = F.relu(self.l2(hidden_layer_1))
        hidden_layer_3 = F.relu(self.l3(hidden_layer_2))
        output_layer = self.out(hidden_layer_3)
        return output_layer

model = MyNetwork()

##########################
# 4. Select an Optimizer #
##########################

from chainer import optimizers
# Choose an optimizer algorithm
optimizer = optimizers.Adam(alpha=0.01) #alpha = learning rate or step size
# Give the optimizer a reference to the model so that it
# can locate the model's parameters.
optimizer.setup(model)

##########################
# 5. Write training loop #
##########################
import numpy as np
from chainer.dataset import concat_examples
# from chainer.cuda import to_cpu

def train_model(model, train_iter, test_iter, max_epoch = 10):
    while train_iter.epoch < max_epoch:
        # ---------- One iteration of the training loop ----------
        train_batch = train_iter.next()
        image_train, target_train = concat_examples(train_batch, device=None) #device is CPU default, but we can use GPU
        # Concatenates a list of examples into array(s).

        # Calculate the prediction of the network
        prediction_train = model(image_train)

        # Calculate the loss with softmax_cross_entropy
        loss = F.softmax_cross_entropy(prediction_train, target_train)

        # Calculate the gradients in the network
        model.cleargrads()
        loss.backward()

        # Update all the trainable paremters
        optimizer.update()
        # --------------------- until here ---------------------

        # Check the validation accuracy of prediction after every epoch
        if train_iter.is_new_epoch:  # If this iteration is the final iteration of the current epoch

            # Display the training loss
            print('epoch:{:02d} train_loss:{:.04f} '.format(
                train_iter.epoch, float(loss.data)), end='')

            test_losses = []
            test_accuracies = []
            while True:
                test_batch = test_iter.next()
                image_test, target_test = concat_examples(test_batch)

                # Forward the test data
                prediction_test = model(image_test)

                # Calculate the loss
                loss_test = F.softmax_cross_entropy(prediction_test, target_test)
                test_losses.append(loss_test.data)

                # Calculate the accuracy
                accuracy = F.accuracy(prediction_test, target_test)
                test_accuracies.append(accuracy.data)

                if test_iter.is_new_epoch:
                    test_iter.epoch = 0
                    test_iter.current_position = 0
                    test_iter.is_new_epoch = False
                    test_iter._pushed_position = None
                    break

            print('val_loss:{:.04f} val_accuracy:{:.04f}'.format(
                np.mean(test_losses), np.mean(test_accuracies)))

#############################
# 6. Save the trained model #
#############################
from chainer import serializers

serializers.save_npz('chainer_mnist_3layer_nn.model', model)

################################################
# 7. Perform classification by the saved model #
################################################
# After training and saving, loading is:
from chainer import serializers

# Define model shape first anyway
model = MyNetwork()
# Load the saved paremeters into the instance
serializers.load_npz('chainer_mnist_3layer_nn.model', model)

# Get a test image and label
x, t = test[0]
plt.imshow(x.reshape(28, 28), cmap='gray')
plt.savefig('7.png')
print('label:', t)

# Change the shape of the minibatch.
# In this example, the size of minibatch is 1.
# Inference using any mini-batch size can be performed.

print(x.shape, end=' -> ')
x = x[None, ...]
print(x.shape)
# (784,) -> (1, 784)

# forward calculation of the model by sending X
y = model(x)

# The result is given as Variable, then we can take a look at the contents by the attribute, .data.
y = y.data

# Look up the most probable digit number using argmax
pred_label = y.argmax(axis=1)

print('predicted label:', pred_label[0])
# predicted label: 7

