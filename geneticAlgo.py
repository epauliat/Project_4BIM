import numpy as np
import matplotlib.pyplot as plt


#############################
######### Functions #########
#############################

# Random genome generation function
def generate_array(size, max = 100, min = 0):
    """Function that generates an array of float comprised between max and min
    Args:
        size (int): the length of the array to be generated
        max (float): Maximal value
        min (float): Minimal value
    Returns:
        numpy.ndarray: array_
    """
    array_ = []
    for i in range(size):
        num = int(np.random.random()*(max-min)+min)
        array_.append(num)
    return np.array(array_)


def generate_population(N, size, max = 100, min = 0):
    """Function that generates an array of floats comprised between max and min
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
    cost = np.zeros(len(arrayList_))
    for i in range(len(arrayList_)):
        cost[i] = cost_function(arrayList_[i], target)
    return cost


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

def array_mutation(array_, P):
    """Function that mutates an array randomly
    Args:
        array_ (numpy.ndarray): The array to be mutated
        P (float): Mutation probability
    Returns:
        numpy.ndarray: newArray_
    """
    S = len(array_)
    newArray_ = np.copy(array_)
    for i in range(S):
        if np.random.random()<P:
            newArray_[i] = array_[i] + np.random.normal(0, 2, 1)
    return newArray_


def crossing_over(arrayList_, P):

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

import random

def coupures(size, N):
    """Computes indexes of the cuts to be performed for the crossing-over
    Args:
        size (int): Length of the arrays
        N (int): Number of fragments
    Returns:
        numpy.ndarray: points
    """
    div = size/N
    points = [int(k*div) for k in range(1,N)]
    return points



def multi_point_crossover(parents):
    """Cette fonction créé une liste de vecteurs enfants en recombinant les parents avec les différents points de coupure

    Args:
        parents : liste des vecteurs après mutation
    Returns:
        childs: un vecteur enfant de la même taille que ceux des parents
    """
    n = len(parents[0])
    points = coupures(n, 5)
    for i in range(len(points)):
        start = points[i]
        childs = np.copy(parents)
        for i in range(len(parents)):
            childs[i][start:] = parents[i+1][start:] if i+1<len(parents) else parents[0][start:]
        parents = childs
    return childs



def liste_mutants_select(vect_select):
    """renvoie la liste de tous les vecteurs mutés à partir des vecteurs séléctionnés
    Args:
        vect_select (_array_): les vecteurs récupérés de la sélection de l'utilisateur dans un array (un array d'array)
    return
        mutants_select (_array_): les vecteurs récupérés de la sélection de l'utilisateur mutés, dans un array (un array d'array)
    """
    mutants_select=np.copy(vect_select)
    for i in range(len(vect_select)-1): #pour chaque vecteur de mon array
        mutants_select[i]=array_mutation(vect_select[i])

    return mutants_select

def mutants_complets(vect_select,mutants_select):
    """calcule le nb mutants manquants s'il y en a et complète la liste avec d'autres mutants (on veut la taille de la liste de mutants=taille de la liste des crossover)
        hypothèse : l'utilisateur ne peut pas sélectionner plus de la moitié des images proposées
    Args:
        vect_select (_array_): les vecteurs récupérés de la sélection de l'utilisateur dans un array
        mutants_select (_array_): les vecteurs récupérés de la sélection de l'utilisateur déjà mutés dans un array
    Returns:
        numpy.ndarray: newcompleteArray_ : tous les vecteurs mutés, dans un array de 5
    """
    S=len(mutants_select)
    #s'il y a 10 vecteurs (images affichées à l'écran) on veut 5 mutants (et 5 modif en crossingover)
    newcompleteArray_= np.copy(mutants_select) #on copie les vecteurs déjà mutés
    #définir newmutant qui sera un nouveau vecteur muté
    new_mutant = np.array([]) #non mutant c'est un vecteur vec_select un vect de vect
    if S<5 :
        nb_mut_manquants=5-S #10=len(vect_select) = taille de la liste des vecteurs affichés
        for i in range (nb_mut_manquants-1):
            new_mutant=array_mutation(vect_select[i])
            #newcompleteArray_.append(new_mutant)
            #t_2 = np.vstack([t_2,ajout_ligne_t_2])
            newcompleteArray_=np.append([newcompleteArray_], [new_mutant])

    elif S==5:
        newcompleteArray_= np.copy(mutants_select)

    return newcompleteArray_

def newGeneration(population_, target, select = .5):
    """Generates a new population by performing mutations on
    a given population
    Args:
        population_ (numpy.ndarray): The array to be mutated
        select (int): Mutation factor
    Returns:
        numpy.ndarray: newPopulation_
    """
    population_ = select_Arrays(population_, target, select)
    newPopulation_ = []

    for array in population_:
        newArray_ = array_mutation(array, 0.5)
        newPopulation_.append(newArray_)

    cross = crossing_over(newPopulation_, 0.5)
    newPopulation_ = np.vstack((newPopulation_, cross))

    return newPopulation_

#############################
######### Main/Test #########
#############################

N = 15
size = 15
min = 0
max = 100

Arr = generate_array(size, max, min)
print(Arr)
print()
mutArr = array_mutation(Arr, 2)
print(mutArr)

pop = generate_population(N, size, max, min)
print(pop)

pop2 = newGeneration(pop, Arr)



print()
print("########################")
print("########################")

print()
pop = generate_population(N, size, max, min)

print()
print("target : ",Arr)

means = []
for i in range(60):
    print()

    pop = newGeneration(pop, Arr)


    print()

    print("########################")
    print("GEN ", i)
    print(len(pop))
    print()
    print("target : ",Arr)
    m = []
    for i in range(len(pop)):
        m.append(cost_function(pop[i], Arr))
    means.append(sum(m)/len(m))

print("######")
print(pop)
print(Arr)
print("######")

print(means)
plt.figure()
plt.plot(np.array(means))
plt.show()

'''
A = [f"A{i}" for i in range(10)]
B = [f"B{i}" for i in range(10)]
C = [f"C{i}" for i in range(10)]
D = [f"D{i}" for i in range(10)]
E = [f"E{i}" for i in range(10)]

for i in [A,B,C,D,E]:
    print(i)

child = multi_point_crossover([A,B,C,D,E])
for i in child:
    print(i)



print("######################")

'''
