from matplotlib import pyplot as plt
import numpy as np
import random
import libs.utils as utils
import pandas as pd
import turicreate as tc

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

data2 = data = tc.SFrame('Hyderabad.csv')

# Building a linear regression model that uses only the "Area" feature
def linear_regression_3():
    simple_model = tc.linear_regression.create(data, features=['Area'], target='Price')
    b, m = simple_model.coefficients['value']
    print("slope:", m)
    print("y-intercept:", b)

    house = tc.SFrame({'Area': [1000]})
    print(f"The predicted price of house with area 1000 is {simple_model.predict(house)[0]}")

    axis[0].scatter(data['Area'], data['Price'])
    axis[1].scatter(data['Area'], data['Price'])
    utils.draw_linear_line(axis[1], m, b, starting=0, ending=max(data['Area']))

    plt.savefig("mygraph.png")

# Building a linear regression model that uses all the features present in the csv file
def linear_regression_4():
    model = tc.linear_regression.create(data, target='Price')
    # print(model.coefficients)
    # print(model.evaluate(data))
    house = tc.SFrame({'Area': [1000], 'No. of Bedrooms':[3]})
    print(f"The predicted price of house with area 1000 and no.of bedrooms 3 is {model.predict(house)[0]}")

    # axis[0].scatter(data['Area'], data['Price'])
    # axis[1].scatter(data['Area'], data['Price'])
    # utils.draw_linear_line(axis[1], m, b, starting=0, ending=max(data['Area']))

# linear_regression_1(features, labels, learning_rate = 0.01, epochs = 10000)
# linear_regression_2(features, labels, learning_rate = 0.01, epochs = 10000)
# linear_regression_3()
# linear_regression_4()