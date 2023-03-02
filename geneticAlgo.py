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


def cost_function(array_, target):
    """Function that computes the distance between an array and a specified target array
    Args:
        array_ (numpy.ndarray): array
        array_ (numpy.ndarray): array
    Returns:
        int: cost
    """
    cost = 0
    for i in array:
        cost += (array_[i] - target[i])**2
    return cost





def select_Arrays(P, Ns):



#############################
######### Main/Test #########
#############################

size = 20
min = -10
max = 15

Arr = generate_array(size, max, min)
print(Arr)
