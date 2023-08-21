# Continuous Perceptron
from matplotlib import pyplot as plt
import numpy as np
import random

import sys
# https://favtutor.com/blogs/import-from-parent-directory-python
sys.path.append('..')
import libs.utils as utils

import turicreate as tc

features = np.array([ [1,0], [0,2], [1,1], [1,2], [1,3], [2,2], [3,2], [2,3]])
labels = np.array([		0,     0,     0,     0,     1,     1,     1,     1 ])

figure, axis = plt.subplots(1,2)

def sigmoid(x):
    # Note, in the book it appears as 1/(1+np.exp(-x)). Both expressions are equivalent, but the expression
    # below behaves better with small floating point numbers.
    return np.exp(x)/(1+np.exp(x))

def score(weights, bias, features):
    return np.dot(weights, features) + bias

# The prediction is the sigmoid activation function applied to the score
def prediction(weights, bias, features):
    return sigmoid(score(weights, bias, features))

def log_loss(weights, bias, features, label):
    pred = 1.0*prediction(weights, bias, features)
    return -label*np.log(pred) - (1-label)*np.log(1-pred)

# We need the log loss over the whole dataset, so we can add over all the data points as shown here
def total_log_loss(weights, bias, features, labels):
    total_error = 0
    for i in range(len(features)):
        total_error += log_loss(weights, bias, features[i], labels[i])
    return total_error

def logistic_trick(weights, bias, features, label, learning_rate = 0.01):
    pred = prediction(weights, bias, features)

    for i in range(len(weights)):
        weights[i] += (label-pred)*features[i]*learning_rate
    
    bias += (label-pred)*learning_rate
    return weights, bias

def logistic_regression_algorithm(features, labels, learning_rate = 0.01, epochs = 1000):
    
    weights = [1.0 for i in range(len(features[0]))]
    bias = 0.0
    errors = []

    for i in range(epochs):
        # Comment the following line of code to remove the plots of all the classifiers
        # utils.draw_line(weights[0], weights[1], bias, color='grey', linewidth=0.1, linestyle='dotted')

        errors.append(total_log_loss(weights, bias, features, labels))
        j = random.randint(0, len(features) - 1)
        weights, bias = logistic_trick(weights, bias, features[j], labels[j])

    utils.plot_points_1(axis[0], features, labels, 'aack', 'beep')
    utils.draw_linear_line_1(axis[0], weights[0], weights[1], bias)
    
    axis[1].scatter(range(epochs), errors)
    axis[1].set(xlabel='epochs')
    axis[1].set(ylabel='errors')

    plt.savefig('mygraph.png')

    return weights, bias

def classifier_1():
    datadict = {'x1': features[:,0], 'x2': features[:,1], 'y': labels}
    data = tc.SFrame(datadict)

    figure, axis = plt.subplots()

    classifier = tc.logistic_classifier.create(data,
                                           features = ['x1', 'x2'],
                                           target = 'y',
                                           validation_set= None)
    # print(classifier.coefficients)

    intercept, w1, w2 = classifier.coefficients["value"]
    # print(a, " ---- ", b, " ------ ", c)

    utils.draw_linear_line_1(axis, w1, w2, intercept)
    utils.plot_points_1(axis, features, labels, 'aack', 'beep')
    plt.savefig('mygraph.png')

# logistic_regression_algorithm(features, labels)
classifier_1()