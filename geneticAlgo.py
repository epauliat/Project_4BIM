import numpy as np
import matplotlib.pyplot as plt


#############################
######### Functions #########
#############################

# Random genome generation function
def generate_array(size, max = 100, min = 0):
    """Function that generates an array of floats comprised between max and min
    Args:
        size (int): the length of the array to be generated
        max (int): Maximal value
        min (int): Minimal value
    Returns:
        numpy.ndarray: array_
    """

    array_ = []
    for i in range(size):
        num = np.random.random()*(max-min)+min
        array_.append(num)
    return np.array(array_)


#############################
######### Main/Test #########
#############################

size = 20
min = -10
max = 15

Arr = generate_array(size, max, min)
print(Arr)
