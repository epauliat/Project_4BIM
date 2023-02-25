import os
import tensorflow as tf
from matplotlib import pyplot as plt
from matplotlib import image as mpimg


def Dataset_Visualisation():
    """Function that prints in seperate popups the images contained in the faces repository
    Args:
        None
    Returns:
        None
    """

    plt.xlabel("X pixel scaling")
    plt.ylabel("Y pixels scaling")

    files = os.listdir('faces')
    for name in files:
        print(name)
        picture = os.listdir('faces/'+name)
        for p in picture:
                # image = mpimg.imread('faces/'+name+"/"+p)
                # plt.title(name)
                # plt.imshow(image)
                # plt.show()
                Image_Visualisation('faces/'+name+"/"+p)

def Image_Visualisation(path):
    """Function that prints in a seperate popup the image whose path is given as an argument
    Args:
        path (str): the image path from the working directory
    Returns:
        None
    """
    image = mpimg.imread(path)
    plt.title(path.split("/")[-1])
    plt.imshow(image)
    plt.show()

def Image_Conversion_to_array(path):
    """Function that converts an image to an array, using a PIL format
    Args:
        path (str): the image path from the working directory
    Returns:
        numpy.ndarray: image_array
    """

    image_pil=tf.keras.utils.load_img(path)
    image_array=tf.keras.utils.img_to_array(image_pil)
    print("array\n")
    print(image_array)
    return image_array


if __name__ == "__main__":
    # Dataset_Visualisation()
    Image_Conversion_to_array("faces/Aaron_Guiel/Aaron_Guiel_0001.jpg")


# def encoder():


# class encoder:
#
#    def __init__(self, images):
#         self.images
