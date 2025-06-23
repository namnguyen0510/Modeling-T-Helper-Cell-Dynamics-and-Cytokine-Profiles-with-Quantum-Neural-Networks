import numpy as np


def MinMaxNormalization(x):
    x = (x-x.min())/(x.max()-x.min())
    return x