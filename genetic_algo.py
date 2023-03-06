!pip install numpy
!pip install pillow
!pip install matplotlib

import random
import numpy as np
from PIL import Image


#on créé une classe pour représenter chaque individu de la population de la banque d'image
class Individual:
    def __init__(self, image_size, num_images): 
        self.chromosome = np.zeros((num_images,)) #np.zeros return a new array of the size of num_images, filled with zeros
        self.image_size = image_size
        self.num_images = num_images
        self.fitness = 0.0
        """Function that initiates the class
    Args:
        image_size (2-tuple containing (width, height) in pixels): size of image from the image bank
        num_images (int): number of images in the image bank
    Returns:
        /
    """
    def randomize(self):
        """Function that randomizes a binary vector chromosome, from the size of the num_images
    Args:
        self
    Returns:
        /
    """
        for i in range(self.num_images):
            self.chromosome[i] = random.randint(0, 1)
    
    def decode(self, image_bank):
        """Function that creates an image by assemblating images from the image bank selected by the chromosome
    Args:
        self
        image_bank=file of bank of images 
    Returns:
        image of the size of image_size
    """ 
        image = Image.new('RGB', self.image_size) #creates a new image of mode RGB and same size as the other
        pixels = image.load() #Allocates storage for the image and loads the pixel data
        for i in range(self.num_images):
            if self.chromosome[i] == 1:
                img = Image.open(image_bank[i])
                img = img.resize((self.image_size[0]//self.num_images, self.image_size[1])) #Returns a resized copy of this image 
                #on la modifie de la taille 
                pixels.paste(img, (i*(self.image_size[0]//self.num_images), 0))
        return image
    
    def evaluate_fitness(self, target_image):
        """La méthode evaluate_fitness calcule la similarité entre l'image créée et l'image cible (à optimiser).
    """
        decoded_image = self.decode(target_image)
        diff = np.asarray(target_image) - np.asarray(decoded_image)
        self.fitness = 1.0 / (1.0 + np.sqrt(np.mean(diff**2)))




#ensuite on créé une classe qui représente la population
class Population:
    def __init__(self, pop_size, image_size, num_images, image_bank, mutation_rate):
        self.pop_size = pop_size
        self.image_size = image_size
        self.num_images = num_images
        self.image_bank = image_bank
        self.mutation_rate = mutation_rate
        self.population = [Individual(image_size, num_images) for i in range(pop_size)]
        for i in range(pop_size):
            self.population[i].randomize()
        
    def selection(self):
        sorted_pop = sorted(self.population, key=lambda x: x.fitness, reverse=True)
        parents = sorted_pop[:2]
        return parents[0], parents[1]
    
    def crossover(self, parent1, parent2):
        child1 = Individual(self.image_size, self.num_images)
        child2 = Individual(self.image_size, self.num_images)
        crossover_point = random.randint(0, self.num_images-1)
        child1.chromosome[:crossover_point] = parent1.chromosome[:crossover_point]
        child1.chromosome[crossover_point:] = parent2.chromosome[crossover_point:]
        child2.chromosome[:crossover_point] = parent2.chromosome[:crossover_point]
        child2.chromosome[crossover_point:] = parent1.chromosome[crossover
