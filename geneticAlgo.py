import numpy as np
import random
import matplotlib.pyplot as plt

# Getting the information regarding the standard deviations of the encoded vectors
stds = []
with open("stds_of_all_encoded_vector_per_position.txt") as f:
    std = f.readlines()
stds = std[0].split(' ')

for i in range(len(stds)):
    stds[i]=float(stds[i])


means = []
with open("means_of_all_encoded_vector_per_position.txt") as f:
    mean = f.readlines()
means = mean[0].split(' ')

for i in range(len(stds)):
    means[i]=float(means[i])


###############################################
################## Functions ##################
###############################################

############################
# Array generation functions

def generate_array(size):
    """Function that generates an array of floats comprised between
    max and min
    Args:
        size (int): the length of the array to be generated
        max (int): Maximal value
        min (int): Minimal value
    Returns:
        numpy.ndarray: array_
    """
    array_ = []
    for i in range(size):
        num = np.random.standard_normal()*stds[i] + means[i] # Random float
        array_.append(num)
    return np.array(array_)


def generate_population(N, size):
    """Function that generates apopulation of arrays of integers
    comprised between max and min
    Args:
        N (int): Number of arrays to generate
        size (int): the length of the arrays to be generated
        max (int): Maximal value
        min (int): Minimal value
    Returns:
        numpy.ndarray: population_
    """
    population_ = []
    for i in range(N):
        population_.append(generate_array(size))

    return np.array(population_)

#################################################
# Cost and selection functions (used for testing)

def cost_function(array_, target):
    """Function that computes the distance between an array and a specified target array

        Args:
            array_ (numpy.ndarray): array
            target (numpy.ndarray): target array
        Returns:
            int
    """

    cost = 0
    diff = target - array_
    for i in range(len(array_)):
        C = np.sqrt(np.mean(diff**2))
        cost += C
    return cost

def cost_population(arrayList_, target):
    """Function that computes the distance between each array of arrayList_
    and a specified target array

        Args:
            arrayList_ (numpy.ndarray): List of arrays to evaluate
            target (numpy.ndarray): target array
        Returns:
            numpy.ndarray
    """
    costs = np.zeros(len(arrayList_))
    for i in range(len(arrayList_)):
        costs[i] = cost_function(arrayList_[i], target)
    return costs


def select_Arrays(arrayList, target, p = 0.5):
    """Function that selects the best arrays based on distance
    to the target array

        Args:
            arrayList (list)
            target (numpy.ndarray): target array
            p (int): proportion of arrays to be selected
        Returns:
            list
    """
    cost = cost_population(arrayList, target)
    idx = np.argsort(cost)
    arrayList = np.array(arrayList)
    orderedArrays = arrayList[idx]

    return orderedArrays[:int(len(arrayList)*p)]


#########################
# Crossing over functions

def crossing_over(arrayList_):
    """Function that performs crossing overs between arrays contained
    in arrayList_. The rate of crossing over is determined by the
    probability P.

        Args:
            arrayList_ (numpy.ndarray): List of arrays to modify by crossing overs
            P (float): probability of a crossing over to occur
        Returns:
            numpy.ndarray
    """
    newArrayList_ = np.copy(arrayList_)
    Nb_arrays, Len_array = newArrayList_.shape

    for i in range(Nb_arrays):
        pos = np.random.randint(Len_array/4, 3*Len_array/4)
        newArrayList_[i] = np.append(arrayList_[i-1][:pos], arrayList_[i][pos:])

    return newArrayList_


def cuts(size, N):
    """Computes indexes of the cuts to be performed for the crossing over
    (multi_point_crossover)

        Args:
            size (int): Length of the arrays
            N (int): Number of fragments
        Returns:
            numpy.ndarray
    """
    div = size/N
    points = [int(k*div) for k in range(1,N)]
    return points


def multi_point_crossover(arrayList_):
    """Function that performs crossing overs between arrays contained
    in arrayList_.
        Args:
            arrayList_ (numpy.ndarray): List of arrays to modify by crossing overs
        Returns:
            numpy.ndarray
    """
    size = len(arrayList_[0])
    N = len(arrayList_)
    points = cuts(size, 5)
    for i in range(len(points)):
        start = points[i]
        newArrayList_ = np.copy(arrayList_)
        for i in range(N):
            newArrayList_[i][start:] = arrayList_[i+1][start:] if i+1<N else arrayList_[0][start:]
        arrayList_ = newArrayList_
    return newArrayList_


###################
# Mutation function

def array_mutation(array_, P):
    """Function that mutates an array randomly

        Args:
            array_ (numpy.ndarray): The array to be mutated
            P (int): Mutation factor
        Returns:
            numpy.ndarray
    """
    max = 8
    min = -8
    S = len(array_)
    mutatedArray_ = np.copy(array_)
    for i in range(S):
        if stds[i] != 0:
            mutatedArray_[i] = array_[i] + np.random.normal(0, P*stds[i])
        if array_[i] > max:
            mutatedArray_[i] = max
        if array_[i] < min:
            mutatedArray_[i] = min

    return mutatedArray_



def list_select_mutants(vect_select,P):
    """Function that mutates randomly all the vectors selected, returns all in a list

        Args:
            vect_select (list): vectors selected by the user in a list
            P (float): Mutation factor
        return
            list
    """
    mutants_select=vect_select.copy()

    for i in range(len(vect_select)): #for each vector of the list vect_select
        mutants_select[i]=array_mutation(vect_select[i],P) #the vector is mutated with the function array_mutation

    return mutants_select


def completes_mutants(vect_select,mutants_select,P):
    """calculate all the missing mutants to complete the list of 5 mutants (to have 5 mutants and 5 vectors by crossing-over)
    --> the user cannot select more than 5 images (10 in total)

        Args:
            vect_select (_list_): vectors selected by the user in a list
            mutants_select (_list_): vectors mutated in a list (from the list of vectos selected by the user)

        Returns:
            numpy.ndarray
    """
    S=len(mutants_select)
    #the size of mutants_select is inferior or equal to 5
    newcompleteMut_=mutants_select.copy() #Copy of the vectors already mutated
    new_mutant = np.array([]) #new_mutant will be a new vector mutated
    if S<5 :
        nb_missing_mut=5-S
        for i in range (nb_missing_mut):
            new_mutant=array_mutation(vect_select[i%len(vect_select)],P)
            newcompleteMut_ = np.append(newcompleteMut_,[new_mutant], axis=0)

    elif S==5:
        newcompleteMut_=mutants_select.copy()

    return newcompleteMut_

def mutatesAll(vect_select,P):
    """_summary_ : function that returns the 5 mutated vectors in a list in once

        Args:
            vect_select (_list_): vectors selected by the user in a list
            P (float): Mutation factor
        Returns :
            numpy.ndarray
        """
    mutants_select=list_select_mutants(vect_select,P)
    newcompleteMut_=completes_mutants(vect_select,mutants_select,P)

    return newcompleteMut_

def allNewvectors(vect_select,P):
    """_summary_ : function that returns the 10 new vectors (5 mutated + 5 crossing over) in a list in once (using all the previous functions)

        Args:
            vect_select (_list_): vectors selected by the user in a list
            P (float): Mutation factor
        Returns :
            numpy.ndarray
        """
    mutated = mutatesAll(vect_select,P)
    print("||||||||||||||||||||||||||||||||||||||", mutated)
    crossedover_vec=multi_point_crossover(mutated)
    allNewvec = np.append(mutated, crossedover_vec, axis = 0)

    return allNewvec



def newGeneration(population_):
    """Generates a new population by performing mutations on
    a given population

        Args:
            population_ (numpy.ndarray): The array to be mutated
            select (int): Mutation factor
        Returns:
            numpy.ndarray
    """
    N = len(population_)
    size = len(population_[0])
    newPopulation_ = []

    if(N==1): # 3 mutated, 3 crossing over, 4 new
        for i in range(3):
            newPopulation_.append(array_mutation(population_[0],1).tolist())
        crossovers = crossing_over(newPopulation_)

        for i in range(3):
            newPopulation_.append(crossovers[i].tolist())

        for i in range(4):
            newPopulation_.append(generate_array(size).tolist())

    elif(N==2): # 4 mutated, 4 crossing over, 2 new
        for i in range(2):
            for j in range(2):
                newPopulation_.append(array_mutation(population_[j],1).tolist())
        crossovers = crossing_over(newPopulation_)

        for i in range(4):
            newPopulation_.append(crossovers[i].tolist())

        for i in range(2):
            newPopulation_.append(generate_array(size).tolist())

    elif(N==3): # 3 mutated, 3 crossing over, 4 new
        for i in range(3):
            newPopulation_.append(array_mutation(population_[i],1).tolist())
        crossovers = crossing_over(newPopulation_)
        print(len(crossovers))

        for i in range(3):
            newPopulation_.append(crossovers[i].tolist())

        for i in range(4):
            newPopulation_.append(generate_array(size).tolist())
        

    elif(N==4): # 4 mutated, 4 crossing over, 2 new
        for i in range(4):
            newPopulation_.append(array_mutation(population_[i],1).tolist())
        crossovers = crossing_over(newPopulation_)

        for i in range(4):
            newPopulation_.append(crossovers[i].tolist())

        for i in range(2):
            newPopulation_.append(generate_array(size).tolist())
        

    elif(N==5): # 5 mutated, 5 crossing over, 0 new
        for i in range(5):
            newPopulation_.append(array_mutation(population_[i],1).tolist())
        crossovers = crossing_over(newPopulation_)

        for i in range(5):
            newPopulation_.append(crossovers[i].tolist())

    return newPopulation_
"""
A = [10 for i in range(10)]
B = [10 for i in range(10)]

new = newGeneration([A])
print(new)
"""
def mutationRate(vect_select):
    """Generates the mutation rate depending on the mean of the difference between max and min of all vect_select
    
        Args:
            vect_select (list): vectors selected by the user in a list
        Returns:
            int
    """

    listDiff = []
    for vect in vect_select:
        listDiff.append(max(vect)-min(vect))

    P = sum(listDiff)/len(listDiff)

    return int(P/3)



#############################
########Main/Test #########
#############################

if __name__ == "__main__":

    print("\n####################################")
    print("Testing of the Genetic Algorithm : \n")

    N = 10

    A = [1100+10*i for i in range(N)]
    B = [2200+10*i for i in range(N)]
    C = [3300+10*i for i in range(N)]
    # D = [450+i for i in range(10)]
    # E = [550+i for i in range(10)]

    stds = [1 for i in range(N)]

    initalPop = np.array([A,B,C])

    print("Initial population : \n", initalPop)

    print("\nMutation on selected arrays...")

    completeMutPop = mutatesAll(initalPop, 1) #Adds mutated arrays until 5 arrays are obtained

    print("\nComplete mutated population : \n", completeMutPop)

    crossovers = multi_point_crossover(completeMutPop)
    print("\nCrossovers obtained : \n", crossovers)

    print("\nThe new generation is then obtained by the concatenation of the complete mutated population and the crossing overs")
    print("\nNew Generation : \n", np.append(completeMutPop, crossovers, axis=0))

#######################################################################
#######################################################################
#######################################################################

    print("\n####################################")
    print("Testing of the Genetic Algorithm : \n")
    print("####################################\n")

    N = 10
    size = 128
    max = 200
    max = -200
    gen = 200
    arrayPop_ = generate_population(N, size)
    arrayTarget_ = generate_array(size)

#######################################################
    def TestFunction(Pop, Target, gen, P, select):
        costs = []
        for i in range(gen):
            cost = cost_population(Pop, Target)
            costs.append(sum(cost)/len(cost))
            Pop = select_Arrays(Pop, Target, .4)
            Pop = allNewvectors(Pop, P)
        return costs


#######################################################
'''
    costs = []
    costs1 = TestFunction(arrayPop_, arrayTarget_, gen, 1, select = 0.5)
    costs2 = TestFunction(arrayPop_, arrayTarget_, gen, 2, select = 0.5)

    t = np.arange(0, gen, 1)


    plt.plot(t, costs1, label = "1")
    plt.plot(t, costs2, label = "2")
    #plt.plot(t, costs3, label = "3")
    plt.legend(loc="upper left")

    plt.ylabel('some number')
    plt.show()
'''

# #############################
# ######### Main/Test #########
# #############################

# '''
# N = 15
# size = 15
# min = 0
# max = 100
#
# Arr = generate_array(size, max, min)
# print(Arr)
# print()
# mutArr = array_mutation(Arr, 2)
# print(mutArr)
#
# pop = generate_population(N, size, max, min)
# print(pop)
#
# pop2 = newGeneration(pop, Arr)
#
#
#
# print()
# print("########################")
# print("########################")
#
# print()
# pop = generate_population(N, size, max, min)
#
# print()
# print("target : ",Arr)
#
# means = []
# for i in range(60):
#     print()
#
#     pop = newGeneration(pop, Arr)
#
#
#     print()
#
#     print("########################")
#     print("GEN ", i)
#     print(len(pop))
#     print()
#     print("target : ",Arr)
#     m = []
#     for i in range(len(pop)):
#         m.append(cost_function(pop[i], Arr))
#     means.append(sum(m)/len(m))
#
# print("######")
# print(pop)
# print(Arr)
# print("######")
#
# print(means)
# plt.figure()
# plt.plot(np.array(means))
# plt.show()
#
#
# A = [f"A{i}" for i in range(10)]
# B = [f"B{i}" for i in range(10)]
# C = [f"C{i}" for i in range(10)]
# D = [f"D{i}" for i in range(10)]
# E = [f"E{i}" for i in range(10)]
#
# for i in [A,B,C,D,E]:
#     print(i)
#
# child = multi_point_crossover([A,B,C,D,E])
# for i in child:
#     print(i)
#
#
#
# print("######################")
#
# A = [10+i for i in range(10)]
# B = [20+i for i in range(10)]
# C = [30+i for i in range(10)]
#D = [f"D{i}" for i in range(10)]
#E = [f"E{i}" for i in range(10)]
#
# print([A,B,C])
#
# stds = [1 for i in range(10)]
#
# E = liste_mutants_select([A,B,C], p, stds)
# #print(E)
# parents=mutants_complets([A,B,C],E)
# print(parents)
# print(multi_point_crossover(parents))
