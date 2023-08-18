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

figure, axis = plt.subplots(1, 3)

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

# Polynomial is -x^2+x+15
coefs = [15,1,-1]

def polynomials(x):
    n = len(coefs)
    return sum([coefs[i]*x**i for i in range(n)])

def create_dataset():
    X = []
    Y = []

    for i in range(40):
        x = random.uniform(-5,5)
        y = polynomials(x) + random.gauss(0,2)
        X.append(x)
        Y.append(y)

    data2 = tc.SFrame({'x':X, 'y':Y})

    for i in range(2,200):
        string = 'x^'+str(i)
        data2[string] = data2['x'].apply(lambda x:x**i)

    return data2

def display_results(plt, model):
    coefs = model.coefficients

    # print("Training error (rmse):", model.evaluate(train)['rmse'])
    # print("Testing error (rmse):", model.evaluate(test)['rmse'])

    plt.scatter(train['x'], train['y'], marker='o')
    plt.scatter(test['x'], test['y'], marker='^')

    utils.draw_polynomial(plt, coefs['value'])

    # print("Polynomial coefficients")
    # print(coefs['name', 'value'])

    plt.figure.savefig("mygraph.png")


figure, axis = plt.subplots()

data2 = create_dataset()
# Our dataset is split into two datasets, the training set called train and the testing set called test.
# A random seed is specified, so we always get the same results, although this is not necessary in practice
train, test = data2.random_split(.8, seed=0)

predictions = test['x', 'y']

def polynomial_regression_1():
    # Training a polynomial regression model with no regularization, here a penalty of 0 means we are not using regularization
    model_no_reg = tc.linear_regression.create(train, target='y', l1_penalty=0.0, l2_penalty=0.0, verbose=False, validation_set=None)
    display_results(axis, model_no_reg)
    
    predictions['No reg'] = model_no_reg.predict(test)
    print(predictions)

def polynomial_regression_2():
    # Training a polynomial regression model with L1 regularization
    model_L1_reg = tc.linear_regression.create(train, target='y', l1_penalty=0.1, l2_penalty=0.0, verbose=False, validation_set=None)
    display_results(axis, model_L1_reg)

    predictions['L1 reg'] = model_L1_reg.predict(test)
    print(predictions)


def polynomial_regression_3():
    # Training a polynomial regression model with L2 regularization
    model_L2_reg = tc.linear_regression.create(train, target='y', l1_penalty=0.0, l2_penalty=0.1, verbose=False, validation_set=None)
    display_results(axis, model_L2_reg)

    predictions['L2 reg'] = model_L2_reg.predict(test)
    print(predictions)

# quadratic_regression_1(features, labels, learning_rate = 0.0001, epochs = 20000)
# quadratic_regression_2(features, labels, learning_rate = 0.0001, epochs = 20000)

# polynomial_regression_1()
# polynomial_regression_2()
# polynomial_regression_3()

