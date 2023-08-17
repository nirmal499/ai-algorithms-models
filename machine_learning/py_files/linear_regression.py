from matplotlib import pyplot as plt
import numpy as np
import random
import libs.utils as utils
import pandas as pd

data = pd.read_csv('Salary_Data.csv')

features = data['YearsExperience']
labels = data['Salary']

min_value_in_features = min(features)
max_value_in_features = max(features)

figure, axis = plt.subplots(1, 2)

# The root mean square error function
def rmse(labels, predictions):
    n = len(labels)
    differences = np.subtract(labels, predictions)
    return np.sqrt(1.0/n * (np.dot(differences, differences)))

# Linear equation: ax^1 + b
def square_trick(a, b, num_rooms, price, learning_rate):
	predicted_price = b + a*num_rooms
	a += learning_rate*num_rooms*(price-predicted_price)
	b += learning_rate*(price-predicted_price)
	return a, b

# Linear equation: ax^1 + b
# Linear Regression Using Square Trick
def linear_regression_1(features, labels, learning_rate=0.01, epochs = 1000):
    a = random.random()
    b = random.random()

    errors = []

    for epoch in range(epochs):

        # utils.draw_linear_line(a, b, starting=0, ending=8)
        # plt.plot(a, b, starting=0, ending=8)

        predictions = a*features[0] + b
        errors.append(rmse(labels, predictions))

        i = random.randint(0, len(features)-1)

        num_rooms = features[i]
        price = labels[i]
        a, b = square_trick(a, b, num_rooms, price, learning_rate=learning_rate)

    # This axis[0] is axis[0,1]
    utils.draw_linear_line(axis[0], a, b, 'black', starting=min_value_in_features, ending=max_value_in_features)

    utils.plot_points(axis[0], features, labels, 'number of rooms', 'prices')

    print('Price per room:', a)
    print('Base price:', b)

    utils.plot_points(axis[1], range(len(errors)), errors)
    
    plt.show()

    return a, b

no_of_data_rows = len(features)

# Linear equation: ax^1 + b
def gradient_descent_step_for_linear_regression(a, b, given_x, given_y, learning_rate):
    predicted_y =  a*given_x + b
    D_a = -2/no_of_data_rows * given_x * (given_y - predicted_y)
    D_b = -2/no_of_data_rows * (given_y - predicted_y)

    a -= learning_rate * D_a
    b -= learning_rate * D_b

    return a,b

def linear_regression_2(features, labels, learning_rate=0.01, epochs = 1000):
    a = random.random()
    b = random.random()

    errors = []

    for epoch in range(epochs):

        # utils.draw_linear_line(a, b, starting=0, ending=8)
        # plt.plot(a, b, starting=0, ending=8)

        predictions = a*features[0] + b
        errors.append(rmse(labels, predictions))

        i = random.randint(0, len(features)-1)

        given_x = features[i]
        given_y = labels[i]
        a, b = gradient_descent_step_for_linear_regression(a, b, given_x, given_y, learning_rate=learning_rate)

    # This axis[0] is axis[0,1]
    utils.draw_linear_line(axis[0], a, b, 'black', starting=min_value_in_features, ending=max_value_in_features)

    utils.plot_points(axis[0], features, labels, 'given_x', 'given_y')

    print('Predicted a in ax + b', a)
    print('Predicted b in ax + b', b)

    utils.plot_points(axis[1], range(len(errors)), errors)
    
    plt.show()

    return a, b


linear_regression_1(features, labels, learning_rate = 0.01, epochs = 10000)
# linear_regression_2(features, labels, learning_rate = 0.01, epochs = 10000)