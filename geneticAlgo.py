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
        mutatedArray_[i] = array_[i] + np.random.randint(-P+1, P)
    return mutatedArray_


def population_mutation(arrayList_,P):
    """Function that mutates each array in the arrayList_ randomly
    Args:
        arrayList_ (numpy.ndarray): The array to be mutated
        P (int): Mutation factor
    Returns:
        numpy.ndarray: mutatedArrayList
    """
    mutatedArrayList_=arrayList_.copy()

    for i in range(len(arrayList_)): #pour chaque vecteur de ma liste vect_select
        mutatedArrayList_[i]=array_mutation(arrayList_[i],P) #je mute ce vecteur avec la fonction array_mutation

    return mutatedArrayList_


def mutants_complets(vect_select,mutants_select,P):
    """calcule le nb mutants manquants s'il y en a et complète la liste avec d'autres mutants (on veut la taille de la liste de mutants=taille de la liste des crossover)
        hypothèse : l'utilisateur ne peut pas sélectionner plus de la moitié des images proposées
    Args:
        vect_select (_liste_): les vecteurs récupérés de la sélection de l'utilisateur dans une liste
        mutants_select (_liste_): les vecteurs récupérés de la sélection de l'utilisateur déjà mutés dans une liste
    Returns:
        numpy.ndarray: newcompleteMut : les 5 vecteurs mutés dans une liste
    """
    S=len(mutants_select)
    #s'il y a 10 vecteurs (images affichées à l'écran) on veut 5 mutants (et 5 modif en crossingover)
    newcompleteMut_=mutants_select.copy() #on copie les vecteurs déjà mutés
    #définir newmutant qui sera un nouveau vecteur muté
    new_mutant = np.array([]) #un nouveau vecteur qui sera muté à partir de la sélection
    if S<5 :
        nb_mut_manquants=5-S #10=len(vect_select) = taille de la liste des vecteurs affichés
        for i in range (nb_mut_manquants):
            new_mutant=array_mutation(vect_select[i],P)
            newcompleteMut_ = np.append(newcompleteMut_,[new_mutant], axis=0)

    elif S==5:
        newcompleteMut_=mutants_select.copy()

    return newcompleteMut_

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

#############################
########Main/Test #########
#############################

if __name__ == "__main__":

    print("\n####################################")
    print("Testing of the Genetic Algorithm : \n")

    A = [150 for i in range(10)]
    B = [250 for i in range(10)]
    C = [350 for i in range(10)]
    D = [450 for i in range(10)]
    E = [550 for i in range(10)]

    initalPop = np.array([A,B,C])
    print("Initial population : \n", initalPop)

    print("\nMutation on selected arrays...")

    mutPop = population_mutation(initalPop, 20) #Mutation of the arrays of initialPop
    print("\nMutated population : \n", mutPop)

    completeMutPop = mutants_complets(initalPop, mutPop, 20) #Adds mutated arrays until 5 arrays are obtained
    print("\nComplete mutated population : \n", completeMutPop)


    crossovers = multi_point_crossover(completeMutPop)
    print("\nCrossovers obtained : \n", crossovers)

    print("\nThe new generation is then obtained by the concatenation of the complete mutated population and the crossing overs")
    print("\nNew Generation : \n", np.append(completeMutPop, crossovers, axis=0))
