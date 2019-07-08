from numpy import *
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import os
import sys


fig = plt.figure()
ax = plt.axes(projection='3d')


# # Data for a three-dimensional line
# zline = linspace(0, 15, 1000)
# xline = sin(zline)
# yline = cos(zline)
# ax.plot3D(xline, yline, zline, 'gray')

# Data for three-dimensional scattered points
# zdata = 15 * random.random(100)
# xdata = sin(zdata) + 0.1 * random.randn(100)
# ydata = cos(zdata) + 0.1 * random.randn(100)
# ax.scatter3D(xdata, ydata, zdata, c=zdata, cmap='Greens')
#
# plt.show()
#
# sys.exit()

def compute_error_for_line_given_points(b, m, points):
    #initialize it at 0
    totalError = 0
    #for every point
    for i in range(0, len(points)):
        #get the x value
        x = points[i, 0]
        #get the y value
        y = points[i, 1]
        #get the difference, square it , add it to the total
        totalError += (y - (m * x + b)) ** 2


    return totalError / float(len(points))


def step_gradient(b_current, m_current, points, learning_rate):

    #starting points for our gradients
    b_gradient = 0
    m_gradient = 0

    n = float(len(points))

    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        #direction with repsect to b and m
        # computing partial derivatives of our error function
        b_gradient += -(2/n) * (y - ((m_current * x) + b_current))
        m_gradient += -(2/n) * x * (y - ((m_current * x) + b_current))

    #update our b and m values using our partial derivatives
    new_b = b_current - (learning_rate * b_gradient)
    new_m = m_current - (learning_rate * m_gradient)
    return [new_b, new_m]

def gradient_descent_runner(points, starting_b, starting_m, learning_rate, num_iterations):
    #starting b and m
    b = starting_b
    m = starting_m

    #gradient descent
    for i in range(num_iterations):
        #update b and m with the new more accurate b and m by performing
        #this gradient step
        b, m = step_gradient(b, m, array(points), learning_rate)

        plt.scatter(0,1)
    plt.show()

    return [b, m]

def run():

    #Step 1 - collect our data
    points = genfromtxt('data.csv', delimiter=',')

    #Step  2 - define our hyperparameters
    #How fast should our model converge?
    learning_rate = 0.0001
    #y = mx + b (slope formula)
    initial_b = 0
    initial_m = 0
    num_iterations = 1000

    #Step 3 - train our model
    print ('starting gradient descent at b = {0}, m = {1}, error = {2}'.format(initial_b, initial_m, compute_error_for_line_given_points(initial_b, initial_m, points)))
    [b, m] = gradient_descent_runner(points, initial_b, initial_m, learning_rate, num_iterations)
    print('After {0} iterations b = {1}, m = {2}, error = {3}'.format(num_iterations, b, m, compute_error_for_line_given_points(b, m, points)))


if __name__ == '__main__':
    run()




# points = genfromtxt('data.csv', delimiter=',')
# points2 = points.tolist()
# print(points2)


# for i in range(0, len(points2)):
#     # plt.scatter(points[i, 0], points[i, 1])
#     Axes3D.scatter(points[i, 0], points[i, 1], 12)000000000
# # plt.axis([0,100,0,100])
# plt.show()
