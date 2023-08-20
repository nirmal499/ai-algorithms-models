import numpy as np
import matplotlib
from matplotlib import pyplot

# y = slope_of_x_2 * x^2 + slope_of_x_1 * x^1 + y_intercept
def draw_quadratic_line(plt, slope_of_x_2, slope_of_x_1, y_intercept,color='grey', linewidth=0.7, starting=0, ending=8):
    x = np.linspace(starting, ending, 1000)
    plt.plot(x, slope_of_x_2*(x**2) +  slope_of_x_1*x + y_intercept, linestyle='-', color=color, linewidth=linewidth)

# y = slope * x^1 + y_intercept
def draw_linear_line(plt, slope, y_intercept, color='grey', linewidth=0.7, starting=0, ending=8):
    x = np.linspace(starting, ending, 1000)
    plt.plot(x, slope*x + y_intercept , linestyle='-', color=color, linewidth=linewidth)

# ax + by + c = 0
# y = -a*(x/b) - (c/b)
def draw_linear_line_1(plt, a,b,c, color='grey', linewidth=0.7, starting=0, ending=3, linestyle='-'):
    x = np.linspace(starting, ending, 1000)
    plt.plot(x, -c/b - a*x/b, linestyle=linestyle, color=color, linewidth=linewidth)

def draw_polynomial(plt, coefs):
    n = len(coefs)
    x = np.linspace(-5, 5, 1000)
    plt.set_ylim([-20,20])
    plt.plot(x, sum([coefs[i]*x**i for i in range(n)]), linestyle='-', color='black')

def plot_points(plt, features, labels, xlabel='',ylabel=''):
    X = np.array(features)
    Y = np.array(labels)
    plt.scatter(X, Y)
    plt.set(xlabel=xlabel)
    plt.set(ylabel=ylabel)

# features: [ [1,0], [0,2], [1,1], [1,2], [1,3], [2,2], [2,3], [3,2] ]
# labels :  [   0,     0,     0,     0,     1,     1,     1,     1   ]
def plot_points_1(plt, features, labels, xlabel='',ylabel=''):
    X = np.array(features)
    y = np.array(labels)

    features_with_label_1 = X[np.argwhere(y == 1)]
    features_with_label_0 = X[np.argwhere(y == 0)]

    plt.scatter([s[0][0] for s in features_with_label_0],
                [s[0][1] for s in features_with_label_0],
                color = 'cyan',
                edgecolor = 'k',
                marker = '^')

    plt.scatter([s[0][0] for s in features_with_label_1],
                [s[0][1] for s in features_with_label_1],
                color = 'red',
                edgecolor = 'k',
                marker = 's')

    plt.set(xlabel=xlabel)
    plt.set(ylabel=ylabel)

    plt.legend(['happy','sad'])
