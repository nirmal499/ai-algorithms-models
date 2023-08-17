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

# Quadratic ax^2 + bx^1 + c
def square_trick(a, b, c, num_rooms, price, learning_rate):
    predicted_price = c + b*num_rooms + a*(num_rooms**2)
    a += learning_rate*(num_rooms**2)*(price-predicted_price)
    b += learning_rate*num_rooms*(price-predicted_price)
    c += learning_rate*(price-predicted_price)

    return a, b, c

# Quadratic ax^2 + bx^1 + c
# Quadratic Regression Using Square Trick
def quadratic_regression_1(features, labels, learning_rate=0.01, epochs = 1000):
    a = random.random()
    b = random.random()
    c = random.random()

    errors = []

    for epoch in range(epochs):

        predictions = a*(features[0]**2) + b*features[0] + c
        errors.append(rmse(labels, predictions))

        i = random.randint(0, len(features)-1)

        num_rooms = features[i]
        price = labels[i]
        a, b, c = square_trick(a,b,c,   num_rooms, 
                                        price, 
                                        learning_rate=learning_rate)
        # print(a, b, c)

    # This axis[0] is axis[0,1]
    utils.draw_quadratic_line(axis[0], a, b, c, 'black', starting=min_value_in_features, ending=max_value_in_features)

    utils.plot_points(axis[0], features, labels, 'number of rooms', 'prices')

    print('a', a)
    print('b', b)
    print('c', c)

    utils.plot_points(axis[1], range(len(errors)), errors)
    print(errors[-1])
    # print(errors)
    
    plt.show()

    return a, b, c

no_of_data_rows = len(features)

# Quadratic equation: ax^2 + bx^1 + c
def gradient_descent_step_for_quadratic_regression(a, b, c, given_x, given_y, learning_rate):
    predicted_y =  a * given_x**2 + b * given_x + c
    D_a = (-2/no_of_data_rows) * given_x**2 * (given_y - predicted_y)
    D_b = (-2/no_of_data_rows) * given_x * (given_y - predicted_y)
    D_c = (-2/no_of_data_rows) * (given_y - predicted_y)

    a -= learning_rate * D_a
    b -= learning_rate * D_b
    c -= learning_rate * D_c

    return a,b,c

def quadratic_regression_2(features, labels, learning_rate=0.01, epochs = 1000):
    a = random.random()
    b = random.random()
    c = random.random()

    errors = []

    for epoch in range(epochs):

        predictions = a*(features[0]**2) + b*features[0] + c
        errors.append(rmse(labels, predictions))

        i = random.randint(0, len(features)-1)

        given_x = features[i]
        given_y = labels[i]
        a, b, c = gradient_descent_step_for_quadratic_regression(a,b,c,   given_x, 
                                        given_y, 
                                        learning_rate=learning_rate)
        # print(a, b, c)

    # This axis[0] is axis[0,1]
    utils.draw_quadratic_line(axis[0], a, b, c, 'black', starting=min_value_in_features, ending=max_value_in_features)

    utils.plot_points(axis[0], features, labels, 'given_x', 'given_y')

    print('Predicted a in ax^2 + bx + c', a)
    print('Predicted b in ax^2 + bx + c', b)
    print('Predicted c in ax^2 + bx + c', c)

    utils.plot_points(axis[1], range(len(errors)), errors)
    print(errors[-1])
    # print(errors)
    
    plt.show()

    return a, b, c

quadratic_regression_1(features, labels, learning_rate = 0.0001, epochs = 20000)
# quadratic_regression_2(features, labels, learning_rate = 0.0001, epochs = 20000)