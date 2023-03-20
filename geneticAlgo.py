import numpy as np
import random
import matplotlib.pyplot as plt


###############################################
################## Functions ##################
###############################################

############################
# Array generation functions

def generate_array(size, max = 100, min = 0):
    """Function that generates an array of integers comprised between
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
        num = np.random.randint(min, max) # Random integer betweeen min and max
        array_.append(num)
    return np.array(array_)


def generate_population(N, size, max = 100, min = 0):
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
        population_.append(generate_array(size, max, min))

    return np.array(population_)

#################################################
# Cost and selection functions (used for testing)

def cost_function(array_, target):
    """Function that computes the distance between an array and a specified target array
    Args:
        array_ (numpy.ndarray): array
        target (numpy.ndarray): target array
    Returns:
        int: cost
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
        numpy.ndarray: costs
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
        list: arraySelection
    """
    cost = cost_population(arrayList, target)
    idx = np.argsort(cost)
    arrayList = np.array(arrayList)
    orderedArrays = arrayList[idx]

    return orderedArrays[:int(len(arrayList)*p)]


#########################
# Crossing over functions

def crossing_over(arrayList_, P):
    """Function that performs crossing overs between arrays contained
    in arrayList_. The rate of crossing over is determined by the
    probability P.
    Args:
        arrayList_ (numpy.ndarray): List of arrays to modify by crossing overs
        P (float): probability of a crossing over to occur
    Returns:
        numpy.ndarray: newArrayList_
    """
    newArrayList_ = np.copy(arrayList_)
    Nb_arrays, Len_array = newArrayList_.shape

    for i in range(0,Nb_arrays):
        if np.random.random() < P:
            idx = np.random.randint(0, Nb_arrays - 1)
            pos = np.random.randint(0, Len_array - 1)

            tmp = newArrayList_[i,pos:Len_array]
            newArrayList_[i,pos:Len_array] = newArrayList_[idx,pos:Len_array]
            newArrayList_[idx,pos:Len_array] = tmp

    return newArrayList_

def cuts(size, N):
    """Computes indexes of the cuts to be performed for the crossing over
    (multi_point_crossover)
    Args:
        size (int): Length of the arrays
        N (int): Number of fragments
    Returns:
        numpy.ndarray: points
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
        numpy.ndarray: newArrayList_
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
        numpy.ndarray: mutatedArray_
    """
    S = len(array_)
    mutatedArray_ = np.copy(array_)
    for i in range(S):
        mutatedArray_[i] = array_[i] + np.random.choice([-P, P])
    return mutatedArray_



def list_select_mutants(vect_select,P):
    """Function that mutates randomly all the vectors selected, returns all in a list
    Args:
        vect_select (_list_): vectors selected by the user in a list
        P (float): Mutation factor
    return
        mutants_select (_list_): vectors mutated in a list (from the list of vectos selected by the user)
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
        numpy.ndarray: newcompleteMut : 5 vectors mutated in a list
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
            numpy.ndarray: newcompleteMut : 5 vectors mutated in a list
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
            numpy.ndarray: allNewvec : 10 new vectors in a list to show to the user
        """
    mutated = mutatesAll(vect_select,P)
    crossedover_vec=multi_point_crossover(mutated)
    allNewvec = np.append(mutated, crossedover_vec, axis = 0)

    return allNewvec



def newGeneration(population_, target, select = .5):
    """Generates a new population by performing mutations on
    a given population
    Args:
        population_ (numpy.ndarray): The array to be mutated
        select (int): Mutation factor
    Returns:
        numpy.ndarray: newPopulation_
    """
    print(len(population_))
    population_ = select_Arrays(population_, target, select)
    print(len(population_))
    newPopulation_ = []

    for array in population_:
        newArray_ = array_mutation(array, 0.5)
        newPopulation_.append(newArray_)

    cross = crossing_over(newPopulation_, 0.5)
    newPopulation_ = np.vstack((newPopulation_, cross))

    return newPopulation_


def mutationRate(vect_select):

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

    A = [150+i for i in range(10)]
    B = [250+i for i in range(10)]
    C = [350+i for i in range(10)]
    D = [450+i for i in range(10)]
    E = [550+i for i in range(10)]

    p = mutationRate([A,B,C])
    print("p = ", p)

    initalPop = np.array([A])
    print("Initial population : \n", initalPop)

    print("\nMutation on selected arrays...")

    completeMutPop = mutatesAll(initalPop, 20) #Adds mutated arrays until 5 arrays are obtained
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

    costs = []
    costs1 = TestFunction(arrayPop_, arrayTarget_, gen, 2, select = 0.5)
    costs2 = TestFunction(arrayPop_, arrayTarget_, gen, 2, select = 0.3)

    t = np.arange(0, gen, 1)


    plt.plot(t, costs1, label = "1")
    plt.plot(t, costs2, label = "2")
    #plt.plot(t, costs3, label = "3")
    plt.legend(loc="upper left")

    plt.ylabel('some number')
    plt.show()


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
# '''
# A = [10+i for i in range(10)]
# B = [20+i for i in range(10)]
# C = [30+i for i in range(10)]
# #D = [f"D{i}" for i in range(10)]
# #E = [f"E{i}" for i in range(10)]
#
# print([A,B,C])
#
# E = liste_mutants_select([A,B,C],p)
# #print(E)
# parents=mutants_complets([A,B,C],E)
# print(parents)
# print(multi_point_crossover(parents))
#
