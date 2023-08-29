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

class Quadratic_Regression:

    # Single variable quadratic regression
    # Quadratic equation: ax^2 + bx + c
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

    # Quadratic Equation: ax^2 + bx^1 + c
    @staticmethod
    def square_trick(a, b, c, num_rooms, price, learning_rate):
        predicted_price = c + b*num_rooms + a*(num_rooms**2)
        
        a += learning_rate*(num_rooms**2)*(price-predicted_price)
        b += learning_rate*num_rooms*(price-predicted_price)
        c += learning_rate*(price-predicted_price)

        return a, b, c

    # Quadratic ax^2 + bx^1 + c
    # Quadratic Regression Using Square Trick
    def quadratic_regression_1(self, learning_rate, epochs):
        a = random.random()
        b = random.random()
        c = random.random()

        errors = []
        for epoch in range(epochs):

            predictions = a*(features[0]**2) + b*features[0] + c
            errors.append(Quadratic_Regression.rmse(labels, predictions))

            i = random.randint(0, self.__no_of_data_rows-1)

            num_rooms = features[i]
            price = labels[i]

            a, b, c = Quadratic_Regression.square_trick(a, b, c, num_rooms, price, learning_rate=learning_rate)

        # This axis[0] is axis[0,1]
        if self.__flag_for_subplot_ON == True:
            utils.draw_quadratic_line(self.__axis[0], a, b, c, 'black', starting=self.__min_value_in_features, ending=self.__max_value_in_features)
            utils.plot_points(self.__axis[0], features, labels, 'number of rooms', 'prices')
        else:
            utils.draw_quadratic_line(self.__axis, a, b, c, 'black', starting=self.__min_value_in_features, ending=self.__max_value_in_features)
            utils.plot_points(self.__axis, features, labels, 'number of rooms', 'prices')

        print('a', a)
        print('b', b)
        print('c', c)

        if self.__flag_for_subplot_ON == True:
            utils.plot_points(self.__axis[1], range(len(errors)), errors)

        if python_version_minor_number == 8:
            # Becoz for some reason plt.show does not work in this situation
            plt.savefig('mygraph.png')
        else:
            plt.show()

        return a, b, c

    # Qudratic equation: ax^2 + bx + c
    # Returns adjusted weights
    def stochastic_gradient_descent_step_for_quadratic_regression(self, a, b, c, given_x, given_y, learning_rate):
        predicted_y =  a * given_x**2 + b * given_x + c

        D_a = (-2/self.__no_of_data_rows) * given_x**2 * (given_y - predicted_y)
        D_b = (-2/self.__no_of_data_rows) * given_x * (given_y - predicted_y)
        D_c = (-2/self.__no_of_data_rows) * (given_y - predicted_y)

        a -= learning_rate * D_a
        b -= learning_rate * D_b
        c -= learning_rate * D_c

        return a,b,c

    def quadratic_regression_2(self, learning_rate, epochs):

        a = random.random()
        b = random.random()
        c = random.random()

        errors = []
        for epoch in range(epochs):

            predictions = a*(features[0]**2) + b*features[0] + c
            errors.append(Quadratic_Regression.rmse(labels, predictions))

            i = random.randint(0, self.__no_of_data_rows-1)

            given_x = features[i]
            given_y = labels[i]
            a, b, c= self.stochastic_gradient_descent_step_for_quadratic_regression(a, b, c, given_x, given_y, learning_rate=learning_rate)

        # This axis[0] is axis[0,1]
        if self.__flag_for_subplot_ON == True:
            utils.draw_quadratic_line(self.__axis[0], a, b, c, 'black', starting=self.__min_value_in_features, ending=self.__max_value_in_features)
            utils.plot_points(self.__axis[0], features, labels, 'given_x', 'given_y')
        else:
            utils.draw_quadratic_line(self.__axis, a, b, c, 'black', starting=self.__min_value_in_features, ending=self.__max_value_in_features)
            utils.plot_points(self.__axis, features, labels, 'given_x', 'given_y')

        print('Predicted a in ax^2 + bx + c', a)
        print('Predicted b in ax^2 + bx + c', b)
        print('Predicted c in ax^2 + bx + c', c)

        if self.__flag_for_subplot_ON == True:
            utils.plot_points(self.__axis[1], range(len(errors)), errors)

        if python_version_minor_number == 8:
            # Becoz for some reason plt.show does not work in this situation
            plt.savefig('mygraph.png')
        else:
            plt.show()

        return a, b, c

    # Quadratic equation: ax^2 + bx + c
    # Returns adjusted weights
    def batch_gradient_descent_step_for_quadratic_regression(self, a, b, c, learning_rate):

        summation_for_a = 0
        summation_for_b = 0
        summation_for_c = 0

        # for i = 0 to len(self.__features) - 1
        for i in range(len(self.__features)):

            given_x = self.__features[i]
            given_y = self.__labels[i]

            predicted_y =  a*given_x**2 + b*given_x + c

            summation_for_a += given_x**2 * (given_y - predicted_y)
            summation_for_b += given_x * (given_y - predicted_y)
            summation_for_c += (given_y - predicted_y)

        D_a = (-2/self.__no_of_data_rows) * summation_for_a
        D_b = (-2/self.__no_of_data_rows) * summation_for_b
        D_c = (-2/self.__no_of_data_rows) * summation_for_c

        a -= learning_rate * D_a
        b -= learning_rate * D_b
        c -= learning_rate * D_c

        return a,b,c

    def quadratic_regression_3(self, learning_rate, epochs):

        a = random.random()
        b = random.random()
        c = random.random()

        errors = []
        for epoch in range(epochs):

            predictions = a*(features[0]**2) + b*features[0] + c
            errors.append(Quadratic_Regression.rmse(labels, predictions))

            a, b, c = self.batch_gradient_descent_step_for_quadratic_regression(a, b, c, learning_rate=learning_rate)

        # This axis[0] is axis[0,1]
        if self.__flag_for_subplot_ON == True:
            utils.draw_quadratic_line(self.__axis[0], a, b, c, 'black', starting=self.__min_value_in_features, ending=self.__max_value_in_features)
            utils.plot_points(self.__axis[0], features, labels, 'given_x', 'given_y')
        else:
            utils.draw_quadratic_line(self.__axis, a, b, c, 'black', starting=self.__min_value_in_features, ending=self.__max_value_in_features)
            utils.plot_points(self.__axis, features, labels, 'given_x', 'given_y')

        print('Predicted a in ax^2 + bx + c', a)
        print('Predicted b in ax^2 + bx + c', b)
        print('Predicted c in ax^2 + bx + c', c)

        if self.__flag_for_subplot_ON == True:
            utils.plot_points(self.__axis[1], range(len(errors)), errors)

        if python_version_minor_number == 8:
            # Becoz for some reason plt.show does not work in this situation
            plt.savefig('mygraph.png')
        else:
            plt.show()

        return a, b, c

# Polynomial is -x^2+x+15
__coefs = [15,1,-1]

def __polynomials(x):
    n = len(__coefs)
    return sum([__coefs[i]*x**i for i in range(n)])

def __create_dataset():
    X = []
    Y = []

    for i in range(40):
        x = random.uniform(-5,5)
        y = __polynomials(x) + random.gauss(0,2)
        X.append(x)
        Y.append(y)

    data2 = tc.SFrame({'x':X, 'y':Y})

    for i in range(2,200):
        string = 'x^'+str(i)
        data2[string] = data2['x'].apply(lambda x:x**i)

    return data2

def __display_results(plt, model, train, test):
    coefs = model.coefficients

    # print("Training error (rmse):", model.evaluate(train)['rmse'])
    # print("Testing error (rmse):", model.evaluate(test)['rmse'])

    plt.scatter(train['x'], train['y'], marker='o')
    plt.scatter(test['x'], test['y'], marker='^')

    utils.draw_polynomial(plt, coefs['value'])

    # print("Polynomial coefficients")
    # print(coefs['name', 'value'])

    plt.figure.savefig("mygraph.png")

def __polynomial_regression_1(axis, train, test, predictions):
    # Training a polynomial regression model with no regularization, here a penalty of 0 means we are not using regularization
    model_no_reg = tc.linear_regression.create(train, target='y', l1_penalty=0.0, l2_penalty=0.0, verbose=False, validation_set=None)
    __display_results(axis, model_no_reg, train, test)
    
    predictions['No reg'] = model_no_reg.predict(test)
    print(predictions)

def __polynomial_regression_2(axis, train, test, predictions):
    # Training a polynomial regression model with L1 regularization
    model_L1_reg = tc.linear_regression.create(train, target='y', l1_penalty=0.1, l2_penalty=0.0, verbose=False, validation_set=None)
    __display_results(axis, model_L1_reg, train, test)

    predictions['L1 reg'] = model_L1_reg.predict(test)
    print(predictions)


def __polynomial_regression_3(axis, train, test, predictions):
    # Training a polynomial regression model with L2 regularization
    model_L2_reg = tc.linear_regression.create(train, target='y', l1_penalty=0.0, l2_penalty=0.1, verbose=False, validation_set=None)
    __display_results(axis, model_L2_reg, train, test)

    predictions['L2 reg'] = model_L2_reg.predict(test)
    print(predictions)

def __setup(option=0):

    if python_version_minor_number != 8:
        sys.exit("Python version is not 3.8. It means Turi Create is not available.!!!")

    figure, axis = plt.subplots()
    data2 = __create_dataset()
    # Our dataset is split into two datasets, the training set called train and the testing set called test.
    # A random seed is specified, so we always get the same results, although this is not necessary in practice
    train, test = data2.random_split(.8, seed=0)

    predictions = test['x', 'y']

    if option == 1:
        __polynomial_regression_1(axis, train, test, predictions)
    elif option == 2:
        __polynomial_regression_2(axis, train, test, predictions)
    elif option == 3:
        __polynomial_regression_3(axis, train, test, predictions)

    # Notice that the model with no regularization fits the training points really well, but it’s chaotic 
    # and doesn’t fit the testing points well. The model with L1 regularization does OK with both the 
    # training and the testing sets. But the model with L2 regularization does a wonderful job with 
    # both the training and the testing sets and also seems to be the one that really captures the shape 
    # of the data

    # From these polynomials, we see the following:
    # 1. For the model with no regularization, all the coefficients are large. This means the 
    # polynomial is chaotic and not good for making predictions.

    # 2. For the model with L1 regularization, all the coefficients, except for the constant one 
    # (the first one), are tiny—almost 0. This means that for the values close to zero, the 
    # polynomial looks a lot like the horizontal line with equation predicted y = 0.57. This is better than 
    # the previous model but still not great for making predictions.

    # 3. For the model with L2 regularization, the coefficients get smaller as the degree grows but 
    # are still not so small. This gives us a decent polynomial for making predictions.

data = pd.read_csv('Salary_Data.csv')

features = data['YearsExperience']
labels = data['Salary']

QR_model = Quadratic_Regression(features, labels, True)
# QR_model = Quadratic_Regression(features, labels, False)

# QR_model.quadratic_regression_1(0.0001, 10000)
# QR_model.quadratic_regression_2(0.0001, 10000)
# QR_model.quadratic_regression_3(0.0001, 10000)

# __setup(3)
