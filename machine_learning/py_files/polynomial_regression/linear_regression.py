from matplotlib import pyplot as plt
import numpy as np
import random

import sys
# https://favtutor.com/blogs/import-from-parent-directory-python
sys.path.append('..')
import libs.utils as utils

import pandas as pd

# sys.version_info(major=3, minor=11, micro=3, releaselevel='final', serial=0)
python_version_minor_number = sys.version_info[1]
if python_version_minor_number == 8:
    import turicreate as tc

class Linear_Regression:

    # Single variable linear regression
    def __init__(self, feature1_arr, label_arr, flag_for_subplot_ON=True):
        self.__features = feature1_arr
        self.__labels = label_arr

        self.__min_value_in_features = min(self.__features)
        self.__max_value_in_features = max(self.__features)

        self.__flag_for_subplot_ON = flag_for_subplot_ON
        if flag_for_subplot_ON == True:
            _, self.__axis = plt.subplots(1,2)
        else:
            _, self.__axis = plt.subplots()

        self.__no_of_data_rows = len(feature1_arr)


    # The root mean square error function
    @staticmethod
    def rmse(labels, predictions):
        n = len(labels)
        differences = np.subtract(labels, predictions)
        return np.sqrt(1.0/n * (np.dot(differences, differences)))

    # Linear equation: ax^1 + b
    @staticmethod
    def square_trick(a, b, num_rooms, price, learning_rate):
        predicted_price = b + a*num_rooms

        a += learning_rate * num_rooms * (price-predicted_price)
        b += learning_rate * (price-predicted_price)

        return a, b

    # Linear equation: ax^1 + b
    # Linear Regression Using Square Trick
    def linear_regression_1(self, learning_rate, epochs):
        a = random.random()
        b = random.random()

        errors = []
        for epoch in range(epochs):

            """
            if self.__flag_for_subplot_ON == True:
                utils.draw_linear_line(self.__axis, a, b, starting=self.__min_value_in_features, ending=self.__max_value_in_features)
                # plt.plot(a, b, starting=self.__min_value_in_features, ending=self.__max_value_in_features)
            else:
                utils.draw_linear_line(self.__axis, a, b, starting=self.__min_value_in_features, ending=self.__max_value_in_features)
                # plt.plot(a, b, starting=self.__min_value_in_features, ending=self.__max_value_in_features)
            """

            predictions = a*features[0] + b
            errors.append(Linear_Regression.rmse(labels, predictions))

            i = random.randint(0, self.__no_of_data_rows-1)

            num_rooms = features[i]
            price = labels[i]

            a, b = Linear_Regression.square_trick(a, b, num_rooms, price, learning_rate=learning_rate)

        # This axis[0] is axis[0,1]
        if self.__flag_for_subplot_ON == True:
            utils.draw_linear_line(self.__axis[0], a, b, 'black', starting=self.__min_value_in_features, ending=self.__max_value_in_features)
            utils.plot_points(self.__axis[0], features, labels, 'number of rooms', 'prices')
        else:
            utils.draw_linear_line(self.__axis, a, b, 'black', starting=self.__min_value_in_features, ending=self.__max_value_in_features)
            utils.plot_points(self.__axis, features, labels, 'number of rooms', 'prices')

        print('Price per room:', a)
        print('Base price:', b)

        if self.__flag_for_subplot_ON == True:
            utils.plot_points(self.__axis[1], range(len(errors)), errors)

        if python_version_minor_number == 8:
            # Becoz for some reason plt.show does not work in this situation
            plt.savefig('mygraph.png')
        else:
            plt.show()

        return a, b

    # Linear equation: ax^1 + b
    # Returns adjusted weights
    def stochastic_gradient_descent_step_for_linear_regression(self, a, b, given_x, given_y, learning_rate):
        predicted_y =  a*given_x + b
        D_a = (-2/self.__no_of_data_rows) * given_x * (given_y - predicted_y)
        D_b = (-2/self.__no_of_data_rows) * (given_y - predicted_y)

        a -= learning_rate * D_a
        b -= learning_rate * D_b

        return a,b

    def linear_regression_2(self, learning_rate, epochs):

        a = random.random()
        b = random.random()

        print(a, b)

        errors = []
        for epoch in range(epochs):

            """
            if self.__flag_for_subplot_ON == True:
                utils.draw_linear_line(self.__axis, a, b, starting=self.__min_value_in_features, ending=self.__max_value_in_features)
                # plt.plot(a, b, starting=self.__min_value_in_features, ending=self.__max_value_in_features)
            else:
                utils.draw_linear_line(self.__axis, a, b, starting=self.__min_value_in_features, ending=self.__max_value_in_features)
                # plt.plot(a, b, starting=self.__min_value_in_features, ending=self.__max_value_in_features)
            """

            predictions = a*features[0] + b
            errors.append(Linear_Regression.rmse(labels, predictions))

            i = random.randint(0, self.__no_of_data_rows-1)

            given_x = features[i]
            given_y = labels[i]
            a, b = self.stochastic_gradient_descent_step_for_linear_regression(a, b, given_x, given_y, learning_rate=learning_rate)

        # This axis[0] is axis[0,1]
        if self.__flag_for_subplot_ON == True:
            utils.draw_linear_line(self.__axis[0], a, b, 'black', starting=self.__min_value_in_features, ending=self.__max_value_in_features)
            utils.plot_points(self.__axis[0], features, labels, 'given_x', 'given_y')
        else:
            utils.draw_linear_line(self.__axis, a, b, 'black', starting=self.__min_value_in_features, ending=self.__max_value_in_features)
            utils.plot_points(self.__axis, features, labels, 'given_x', 'given_y')

        print('Predicted a in ax + b', a)
        print('Predicted b in ax + b', b)

        if self.__flag_for_subplot_ON == True:
            utils.plot_points(self.__axis[1], range(len(errors)), errors)

        if python_version_minor_number == 8:
            # Becoz for some reason plt.show does not work in this situation
            plt.savefig('mygraph.png')
        else:
            plt.show()

        return a, b

    # Linear equation: ax^1 + b
    # Returns adjusted weights
    def batch_gradient_descent_step_for_linear_regression(self, a, b, learning_rate):

        summation_for_a = 0
        summation_for_b = 0

        # for i = 0 to len(self.__features) - 1
        for i in range(len(self.__features)):

            given_x = self.__features[i]
            given_y = self.__labels[i]

            predicted_y =  a*given_x + b

            summation_for_a += given_x * (given_y - predicted_y)
            summation_for_b += (given_y - predicted_y)

        D_a = (-2/self.__no_of_data_rows) * summation_for_a
        D_b = (-2/self.__no_of_data_rows) * summation_for_b

        a -= learning_rate * D_a
        b -= learning_rate * D_b

        return a,b

    def linear_regression_3(self, learning_rate, epochs):

        a = random.random()
        b = random.random()

        print(a, b)

        errors = []
        for epoch in range(epochs):

            """
            if self.__flag_for_subplot_ON == True:
                utils.draw_linear_line(self.__axis, a, b, starting=self.__min_value_in_features, ending=self.__max_value_in_features)
                # plt.plot(a, b, starting=self.__min_value_in_features, ending=self.__max_value_in_features)
            else:
                utils.draw_linear_line(self.__axis, a, b, starting=self.__min_value_in_features, ending=self.__max_value_in_features)
                # plt.plot(a, b, starting=self.__min_value_in_features, ending=self.__max_value_in_features)
            """

            predictions = a*features[0] + b
            errors.append(Linear_Regression.rmse(labels, predictions))

            a, b = self.batch_gradient_descent_step_for_linear_regression(a, b, learning_rate=learning_rate)

        # This axis[0] is axis[0,1]
        if self.__flag_for_subplot_ON == True:
            utils.draw_linear_line(self.__axis[0], a, b, 'black', starting=self.__min_value_in_features, ending=self.__max_value_in_features)
            utils.plot_points(self.__axis[0], features, labels, 'given_x', 'given_y')
        else:
            utils.draw_linear_line(self.__axis, a, b, 'black', starting=self.__min_value_in_features, ending=self.__max_value_in_features)
            utils.plot_points(self.__axis, features, labels, 'given_x', 'given_y')

        print('Predicted a in ax + b', a)
        print('Predicted b in ax + b', b)

        if self.__flag_for_subplot_ON == True:
            utils.plot_points(self.__axis[1], range(len(errors)), errors)

        if python_version_minor_number == 8:
            # Becoz for some reason plt.show does not work in this situation
            plt.savefig('mygraph.png')
        else:
            plt.show()

        return a, b

# Building a linear regression model that uses only the "Area" feature using turi create
def __linear_regression_3():

    if python_version_minor_number != 8:
        sys.exit("Python version is not 3.8. It means Turi Create is not available.!!!")
    
    data = tc.SFrame('Hyderabad.csv')
    _, axis = plt.subplots()
    
    simple_model = tc.linear_regression.create(data, features=['Area'], target='Price')
    b, m = simple_model.coefficients['value']
    
    print("slope:", m)
    print("y-intercept:", b)

    house = tc.SFrame({'Area': [1000]})
    print(f"The predicted price of house with area 1000 is {simple_model.predict(house)[0]}")

    axis.scatter(data['Area'], data['Price'])
    utils.draw_linear_line(axis, m, b, starting=0, ending=max(data['Area']))

    plt.savefig("mygraph.png")

# Building a linear regression model that uses all the features present in the csv file using turi create
def __linear_regression_4():

    if python_version_minor_number != 8:
        sys.exit("Python version is not 3.8. It means Turi Create is not available.!!!")

    data = tc.SFrame('Hyderabad.csv')
    _, axis = plt.subplots()

    model = tc.linear_regression.create(data, target='Price')
    # print(model.coefficients)
    # print(model.evaluate(data))
    house = tc.SFrame({'Area': [1000], 'No. of Bedrooms':[3]})
    print(f"The predicted price of house with area 1000 and no.of bedrooms 3 is {model.predict(house)[0]}")

    # axis.scatter(data['Area'], data['Price'])
    # utils.draw_linear_line(axis, m, b, starting=0, ending=max(data['Area']))

data = pd.read_csv('placement.csv')

features = data['cgpa']
labels = data['package']

# LR_model = Linear_Regression(features, labels, True)
LR_model = Linear_Regression(features, labels, False)

# LR_model.linear_regression_1(0.01, 10000)

# LR_model.linear_regression_2(0.01, 1000000)
# LR_model.linear_regression_3(0.001, 100000)

# __linear_regression_3()
# __linear_regression_4()