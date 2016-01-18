import random
import numpy as np
from test.linear.ridge import read_data
from linear.ridge import RidgeRegress
from linear.ordinary import LeastSquares


if __name__ == '__main__':
    x_arrays, y_array = read_data()
    # RidgeRegress
    ridge = RidgeRegress(x_arrays, y_array)
    ridge.cross_train()
    print("RidgeRegress's weights are:")
    print(ridge.get_weights())
    # Standard Regress
    for x_array in x_arrays:
        x_array.insert(0, 1.0)
    ordinary = LeastSquares(x_arrays, y_array)
    ordinary.train()
    print("\r\nOrdinaryLastSquares's weights are:")
    print(ordinary.get_weights())












