import os
import sys
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import image as mpimg

from keras_preprocessing.image import img_to_array

import torch
import torch.nn as nn
from torchvision import transforms

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

def Data_import():
    """Function that imports all the images in the file faces, and put them in a list
    Args:
     None
    Returns:
     list: array of all the image as tensors

    """

    files = os.listdir('faces')
    data=[]
    for name in files:
        picture = os.listdir('faces/'+name)
        for p in picture:
            data.append(Image_Conversion_to_tensor("faces/"+name+"/"+p))
    return data

def Image_Conversion_to_tensor(path):
    """Function that converts an image to an array, using a PIL format
    Args:
        path (str): the image path from the working directory
    Returns:
        tensor: the image tensor correspinding to the path
    """

    image_pil=Image.open(path)
    transformation=transforms.ToTensor()
    image_tensor=transformation(image_pil)
    return image_tensor

class Autoencoder(nn.Module):
    def __init__(self):
        """Autoencoder Constructor
        """
        super().__init__()
        # N, 1, 28, 28
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=1, padding=0), # -> N, 16, 14, 14
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=1, padding=0), # -> N, 32, 7, 7
            nn.ReLU(),
            nn.Conv2d(32, 64, 7), # -> N, 64, 1, 1
            nn.Flatten(),
            nn.Linear(3686400,20)


        )

        # N , 64, 1, 1
        self.decoder = nn.Sequential(
            nn.Linear(20,3686400),
            nn.Unflatten(1,[64,240,240]),
            nn.ConvTranspose2d(64, 32, 7), # -> N, 32, 7, 7
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, stride=1, padding=0, output_padding=0), # N, 16, 14, 14 (N,16,13,13 without output_padding)
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, 3, stride=1, padding=0, output_padding=0), # N, 1, 28, 28  (N,1,27,27)
            nn.Sigmoid(),
        )

    def forward(self, x):
        """Function that encodes an an image tensor, then decodes it
        Args:
            x (tensor): the  original image tensor
        Returns:
            tensor: the image tensor after being decoded
        """
        encoded = self.encoder(x)
        # print("here I am")
        decoded = self.decoder(encoded)
        return decoded

def train(data, autoencoder):
    """Function that train an autoencoder with a given dataset
    Args:
        data (list): the  list of image tensors
    Returns:
        None
    """
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=1e-3, weight_decay=1e-5)
    num_epochs = 30
    outputs = []
    i=0
    for epoch in range(num_epochs):
        for img in data:
            img = img.reshape(1, 3, 250, 250) # -> use for Autoencoder_Linear
            print(str(i), end=',')
            sys.stdout.flush()
            i+=1
            recon = autoencoder(img)
            # print("hereeeeeee")
            loss = criterion(recon, img)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        i=0
        print(f'______________Epoch:{epoch+1}, Loss:{loss.item():.4f}')
        outputs.append((epoch, img, recon))

def save(autoencoder, path):
    """Function that saves an autoencoder to a file
    Args:
        autoencoder (Autoencoder): the model that we want to save
    Returns:
        None
    """
    torch.save(autoencoder.state_dict(), path)

def load(path):
    """Function that loads an autoencoder from a file
    Args:
        path (str): the path we want to load from
    Returns:
        autoencoder (Autoencoder): the loaded autoencoder
    """
    autoencoder = Autoencoder()
    autoencoder.load_state_dict(torch.load(path))
    autoencoder.eval()
    return autoencoder

def creating_training_saving_autoencoder(path):
    """Function that creates an autoencoder, trains it, and saves it to a file
    Args:
        None
    Returns:
        autoencoder (Autoencoder): the created autoencoder
    """
    data=Data_import()
    print("I got the data")
    my_autoencoder=Autoencoder()
    print("I got the autoencoder")
    train(data, my_autoencoder)
    save(my_autoencoder, path)
    return my_autoencoder

def loading_autoencoder(path):
    """Function that load an autoencoder
    Args:
        None
    Returns:
        autoencoder (Autoencoder): the created autoencoder
    """
    loaded_autoencopder=load(path)
    return loaded_autoencopder

def comparing_images(autoencoder, path_to_image):
    """Function that shows 2 images: one originale and one that has been
    been coded and decoded by an autoencoder
    Args:
        autoencoder (Autoencoder): the autoencoder used
        path (str): the path we want to load the image from
    Returns:
        None
    """
    image_tensor= Image_Conversion_to_tensor(path_to_image)
    X=image_tensor.reshape(1,3,250,250)
    decoded_tensor=autoencoder.forward(X)
    decoded_pil=transforms.functional.to_pil_image(decoded_tensor.reshape(3,250,250))

    fig = plt.figure(figsize=(50,50))
    fig.add_subplot(1, 2, 1)
    plt.imshow(decoded_pil)
    fig.add_subplot(1, 2, 2)
    plt.imshow(mpimg.imread(path_to_image))
    plt.show()


if __name__ == "__main__":
    # my_autoencoder=creating_training_saving_autoencoder("autoencoder_fitted_85faces_30epochs.pt")
    # comparing_images(my_autoencoder,"faces/Afton_Smith/Afton_Smith_0001.jpg")

    my_autoencoder_loaded=loading_autoencoder("autoencoder_fitted_29faces_10epochs.pt")
    comparing_images(my_autoencoder_loaded,"faces/Azra_Akin/Azra_Akin_0001.jpg")
    comparing_images(my_autoencoder,"faces/Aaron_Patterson/Aaron_Patterson_0001.jpg")

    # comparing_images(my_autoencoder_loaded,"faces/Afton_Smith/Afton_Smith_0001.jpg")
