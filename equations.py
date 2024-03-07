"""
Add your own equations here! Don't forget to import them into main.py
"""


import numpy as np


def equation_1(xx, yy, params):
    return np.sin(xx ** 2 - yy ** 2 + params[0]), np.cos(2 * xx * yy + params[1])


def equation_2(xx, yy, params):
    return np.sin(params[0] * xx) + np.cos(params[1] * yy), np.sin(params[1] * xx) - np.cos(params[0] * yy)



def equation_6(xx, yy, params):
    x = np.sin(xx ** 2 - yy ** 2 + params[0]) + np.cos(2 * xx * yy + params[2])
    y = np.cos(2 * xx * yy + params[1]) - np.sin(xx ** 2 - yy ** 2 + params[2])
    return x, y


def equation_7(xx, yy, params):
    x = np.sin(xx ** 3 - yy ** 2 + params[0]) + np.cos(2 * xx * yy + params[2])
    y = np.cos(2 * xx * yy + params[1]) - np.sin(xx ** 2 - yy ** 3 + params[2])
    return x, y