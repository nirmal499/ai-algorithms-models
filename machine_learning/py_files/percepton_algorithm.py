from matplotlib import pyplot as plt
import numpy as np
import random
import libs.utils as utils
import turicreate as tc

# features[0] -> [1, 0] represents that the word 'aack' appears 1 times and 'beep' appears 0 times
# labels[0] -> 0 represents that [1,0] has label 0 means the sentence is SAD
features = np.array([[1,0],[0,2],[1,1],[1,2],[1,3],[2,2],[2,3],[3,2]])
labels = np.array([0,0,0,0,1,1,1,1])

figure, axis = plt.subplots(1,2)

# figure, axis = plt.subplots()
# utils.plot_points_1(axis, features, labels, 'aack', 'beep')
# plt.savefig('mygraph.png')

# Calculates the dot product between the weights and the features, adds the bias, and applies the step function
def score(weights, bias, features):
    return features.dot(weights) + bias

def step(x):
    if x >= 0:
        return 1
    else:
        return 0

# Here weights is an array
# Here bias is an integer
# Looks at the score, and if it is positive or zero, returns 1; if it is negative, returns 0
def prediction(weights, bias, features):
    return step(score(weights, bias, features))

# If the prediction is equal to the label, then the point is well classified, which means the error is zero.
# If the prediction is different from the label, then the point is misclassified, which means the error is equal to the 
# absolute value of the score.
def error(weights, bias, features, label):
    pred = prediction(weights, bias, features)
    if pred == label:
        return 0
    else:
        return np.abs(score(weights, bias, features))

# Loops through our data, and for each point, adds the error at that point, then returns this error
def mean_perceptron_error(weights, bias, features, labels):
    total_error = 0
    for i in range(len(features)):
        total_error += error(weights, bias, features[i], labels[i])

    # The sum of errors is divided by the number of points to get the mean perceptron error.
    return total_error/len(features)

# We would provide a intial weights and bias
def perceptron_trick(weights, bias, features, label, learning_rate = 0.01):
    pred = prediction(weights, bias, features)

    # Updates the weights and biases using the perceptron trick
    if pred == label:
        return weights, bias
    else:
        if label == 1 and pred == 0:

        	# Predicted label is No and Given label is Yes
        	# So, we need to change the weights

            for i in range(len(weights)):
                weights[i] += features[i]*learning_rate

            bias += learning_rate
        elif label == 0 and pred == 1:

        	# Predicted label is Yes and Given label is No
        	# So, we need to change the weights

            for i in range(len(weights)):
                weights[i] -= features[i]*learning_rate

            bias -= learning_rate
    return weights, bias

random.seed(0)

def perceptron_algorithm(features, labels, learning_rate = 0.01, epochs = 200):
    # Initializes the weights to 1 and the bias to 0. Feel free to initialize them to small random numbers if you prefer.

	# No.of elements in weights define the no.of features. Here we have two: 'No.of times aack appeared'
	# and 'No.of times beep appeared'
    weights = [1.0 for i in range(len(features[0]))] # Initial weights
    bias = 0.0 # Initial bias

    errors = []

    # Repeats the process as many times as the number of epochs
    for epoch in range(epochs):
        # Coment the following line to draw only the final classifier
        # utils.draw_linear_line_1(axis[0], weights[0], weights[1], bias, color='grey', linewidth=1.0, linestyle='dotted')

         # Calculates the mean perceptron error and stores it
        error = mean_perceptron_error(weights, bias, features, labels)
        errors.append(error)

        # Picks a random point in our dataset
        i = random.randint(0, len(features) - 1)

        # Applies the perceptron algorithm to update the weights and the bias of our model based on that point
        weights, bias = perceptron_trick(weights, bias, features[i], labels[i])

    utils.draw_linear_line_1(axis[0], weights[0], weights[1], bias)

    utils.plot_points_1(axis[0], features, labels)

    axis[1].scatter(range(epochs), errors)

    plt.savefig('mygraph.png')

    return weights, bias


def classifier_1():
    datadict = {'aack': features[:,0], 'beep':features[:,1], 'prediction': labels}
    data = tc.SFrame(datadict)

    figure, axis = plt.subplots()

    perceptron = tc.logistic_classifier.create(data, target='prediction')
    # print(perceptron.coefficients)

    a, b, c = perceptron.coefficients["value"]
    # print(a, " ---- ", b, " ------ ", c)

    utils.draw_linear_line_1(axis, a,b,c)
    utils.plot_points_1(axis, features, labels)
    plt.savefig('mygraph.png')

# perceptron_algorithm(features, labels)
# classifier_1()