import numpy as np
import matplotlib
from matplotlib import pyplot

def draw_quadratic_line(plt, slope_of_x_2, slope_of_x_1, y_intercept,color='grey', linewidth=0.7, starting=0, ending=8):
    x = np.linspace(starting, ending, 1000)
    plt.plot(x, slope_of_x_2*(x**2) +  slope_of_x_1*x + y_intercept, linestyle='-', color=color, linewidth=linewidth)

def draw_linear_line(plt, slope, y_intercept, color='grey', linewidth=0.7, starting=0, ending=8):
    x = np.linspace(starting, ending, 1000)
    plt.plot(x, slope*x + y_intercept , linestyle='-', color=color, linewidth=linewidth)

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