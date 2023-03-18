import os
import sys
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import image as mpimg
from torchsummary import summary

from keras_preprocessing.image import img_to_array

import torch
import torch.nn as nn
from torchvision import transforms, datasets

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

def Data_import(path, batchsize):
    """Function that imports all the images in the file faces, and put them in a list
    Args:
     None
    Returns:
     list: array of all the image as tensors
    """
    dataset = datasets.ImageFolder(root=path, transform=transforms.Compose([transforms.Resize((128,128)),transforms.ToTensor()]))
    loader = torch.utils.data.DataLoader(dataset, batch_size = batchsize)
    return loader

def Image_Conversion_to_tensor(path):
    """Function that converts an image to an array, using a PIL format
    Args:
        path (str): theresize((64,64)) image path from the working directory
    Returns:
        tensor: the image tensor correspinding to the path
    """
    image_pil=Image.open(path)
    image_pil_resized=image_pil.resize((128,128))
    transformation=transforms.ToTensor()
    image_tensor=transformation(image_pil_resized)
    return image_tensor

class Encoder(nn.Module):
    def __init__(self):
        """Encoder Constructor
        """
        super().__init__()
        # N, 1, 128, 128
        self.encoder_Conv2d_ReLU_1 = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=1, padding=0),
            nn.ReLU()
        )
        self.encoder_Conv2d_ReLU_2 = nn.Sequential(
            nn.Conv2d(16, 16, 3, stride=1, padding=0),
            nn.ReLU()
        )
        self.encoder_MaxPool2d = nn.MaxPool2d(3) #, return_indices=True
        self.encoder_Flatten = nn.Flatten()
        self.encoder_Linear = nn.Linear(25600,128)
        self.BatchNormalization = nn.BatchNorm2d(16)

    def forward(self, x):
        """Function that encodes an an image tensor
        Args:
            x (tensor): the  original image tensor
        Returns:
            tensor: the image tensor after being encoded
        """

        encoded = self.encoder_Conv2d_ReLU_1(x)
        encoded_pooling = self.encoder_MaxPool2d(encoded) # indices
        # print("1",encoded.size())
        encoded = self.encoder_Conv2d_ReLU_2(encoded)
        encoded_pooling = self.encoder_MaxPool2d(encoded) # indices
        # print("2",encoded.size())
        encoded = self.BatchNormalization(encoded)
        # print("3",encoded.size())
        encoded = self.encoder_Conv2d_ReLU_2(encoded)
        # print("4",encoded.size())
        encoded_pooling = self.encoder_MaxPool2d(encoded) # indices
        # print("5",encoded_pooling.size())
        encoded_flatten = self.encoder_Flatten(encoded_pooling)
        # print("6",encoded_pooling.size())
        encoded_Linear = self.encoder_Linear(encoded_flatten)
        # print("7",encoded_pooling.size())
        return encoded_Linear #, indices

class Decoder(nn.Module):

    def __init__(self):
        """Decoder Constructor
        """
        super().__init__()
        self.BatchNormalization = nn.BatchNorm2d(16)

        self.decoder_Linear = nn.Linear(128,238144)
        self.decoder_Unflatten = nn.Unflatten(1,[16,122,122])
        # self.decoder_MaxUnpool2d = nn.MaxUnpool2d(3)
        self.decoder_ReLu_ConvTranspose2d_2 = nn.Sequential(
            nn.ConvTranspose2d(16, 16, 3, stride=1, padding=0, output_padding=0),
            nn.ReLU()
        )
        self.decoder_ReLu_ConvTranspose2d_1 = nn.Sequential(
            nn.ConvTranspose2d(16, 3, 3, stride=1, padding=0, output_padding=0),
            nn.Sigmoid()
        )


    def forward(self, x): #, indices
        """Function that encodes an an image tensor, then decodes it
        Args:
            x (tensor): the  original image tensor
        Returns:
            tensor: the image tensor after being decoded
        """
        decoded_Unlinear = self.decoder_Linear(x)
        # print("1",decoded_Unlinear.size())
        decoded_unflatten = self.decoder_Unflatten(decoded_Unlinear)
        # print("2",decoded_unflatten.size())
        # decoded_unpooling = self.decoder_MaxUnpool2d(decoded_unflatten, indices, output_size = [1, 26, 58, 58])
        decoded = self.decoder_ReLu_ConvTranspose2d_2(decoded_unflatten)
        # print("3",decoded.size())
        decoded = self.BatchNormalization(decoded)
        # print("4",decoded.size())
        decoded = self.decoder_ReLu_ConvTranspose2d_2(decoded)
        # print("5",decoded.size())
        decoded = self.decoder_ReLu_ConvTranspose2d_1(decoded)
        # print("6",decoded.size())
        return decoded

class Autoencoder(nn.Module):
    def __init__(self):
        """Autoencoder Constructor
        """
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        encoded = self.encoder(x) #, indices
        # print("encoded",encoded.size())
        decoded = self.decoder(encoded) #, indices
        # print("dedcoded",decoded.size())
        return decoded

def train(nb_epoch, data, autoencoder):
    """Function that train an autoencoder with a given dataset
    Args:
        data (list): the  list of image tensors
    Returns:
        None
    """
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=1e-3, weight_decay=1e-5)
    outputs = []
    i=0
    for epoch in range(nb_epoch):
        for (index, batch) in enumerate(data):
            loss=0
            print("batch number "+str(index), end="  :")
            for j in range(len(batch[0])):
                img=batch[0][j]
                batchsize=len(batch[0])
                img = img.reshape(1, 3, 128, 128)
                recon=autoencoder(img)
                # print(recon.size())
                if(i%10==0):
                    print(str(i), end=",")
                sys.stdout.flush()
                i+=1
                loss+=criterion(recon, img)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print("")
        i=0

        print(f'__________Epoch:{epoch+1}, Loss:{loss.item():.4f}')
        outputs.append((epoch, img, recon))

def save(model, path):
    """Function that saves a model to a file
    Args:
        autoencoder (Autoencoder): the model that we want to save
    Returns:
        None
    """

    # print(model.state_dict())
    # print("____________________ SAVE FUNCTION STATE DICT __________________")
    # torch.save(model.state_dict(), path)
    torch.save(model.state_dict(), path)

def save_decoder(model, path):
    """Function that saves a model to a file
    Args:
        autoencoder (Autoencoder): the model that we want to save
    Returns:
        None
    """

    # print(model.state_dict())
    # print("____________________ SAVE FUNCTION STATE DICT __________________")
    # torch.save(model.state_dict(), path)
    torch.save(model.decoder.state_dict(), path)

def save_encoder(model, path):
    """Function that saves a model to a file
    Args:
        autoencoder (Autoencoder): the model that we want to save
    Returns:
        None
    """

    # print(model.state_dict())
    # print("____________________ SAVE FUNCTION STATE DICT __________________")
    # torch.save(model.state_dict(), path)
    torch.save(model.encoder.state_dict(), path)

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

def load_decoder(path):
    """Function that loads an autoencoder from a file
    Args:
        path (str): the path we want to load from
    Returns:
        autoencoder (Autoencoder): the loaded autoencoder
    """
    decoder=Decoder()
    decoder.load_state_dict(torch.load(path))
    decoder.eval()
    return decoder

def load_encoder(path):
    """Function that loads an autoencoder from a file
    Args:
        path (str): the path we want to load from
    Returns:
        autoencoder (Autoencoder): the loaded autoencoder
    """
    encoder=Encoder()
    encoder.load_state_dict(torch.load(path))
    encoder.eval()
    return encoder

def creating_training_saving_autoencoder(nb_epoch, batchsize, saving_path, data_path,saving_path_decoder,saving_path_encoder):
    """Function that creates an autoencoder, trains it, and saves it to a file
    Args:
        None
    Returns:
        autoencoder (Autoencoder): the created autoencoder
    """
    data=Data_import(data_path, batchsize)
    print("I got the data")
    my_autoencoder=Autoencoder()
    print("I got the autoencoder")
    train(nb_epoch, data, my_autoencoder)
    save(my_autoencoder, saving_path)
    save_decoder(my_autoencoder, saving_path_decoder)
    save_encoder(my_autoencoder, saving_path_encoder)
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
    X=image_tensor.reshape(1,3,128,128)
    decoded_tensor=autoencoder.forward(X)
    decoded_pil=transforms.functional.to_pil_image(decoded_tensor.reshape(3,128,128))

    f,axs=plt.subplots(1,2,figsize=(5,5)) #ROW,COLUMN
    axs[0].imshow(decoded_pil)
    axs[1].imshow(mpimg.imread(path_to_image))
    plt.show()

def decoding_images(decoder, encoder, path_to_image):
    """Function that shows 2 images: one originale and one that has been
    been coded and decoded by an autoencoder
    Args:
     autoencoder (Autoencoder): the autoencoder used
     path (str): the path we want to load the image from
    Returns:
     None
    """
    image_tensor= Image_Conversion_to_tensor(path_to_image)
    X=image_tensor.reshape(1,3,128,128)
    encoded_vector=encoder.forward(X) #, indices
    decoded_tensor=decoder.forward(encoded_vector) #, indices
    decoded_pil=transforms.functional.to_pil_image(decoded_tensor.reshape(3,128,128))

    fig = plt.figure(figsize=(20,20))
    fig.add_subplot(1, 2, 1)
    plt.imshow(decoded_pil)
    fig.add_subplot(1, 2, 2)
    plt.imshow(mpimg.imread(path_to_image))
    plt.show()

def encodedVectorGenerator(path, encoder):
    image_tensor= Image_Conversion_to_tensor(path)
    X=image_tensor.reshape(1,3,64,64)
    encoded_vector=encoder.forward(X) #, indices
    return encoded_vector.detach().numpy()

def decoding_Vector(x, decoder):
    decoded_tensor=decoder.forward(torch.tensor(x)) #, indices
    decoded_pil=transforms.functional.to_pil_image(decoded_tensor.reshape(3,64,64))
    return decoded_pil


if __name__ == "__main__":
    epoch = 5
    batch_size = 10
    my_autoencoder=creating_training_saving_autoencoder(epoch, batch_size, "models/autoencoder_18_03.pt",'few_faces',"models/decoder_18_03.pt","models/encoder_18_03.pt")
    comparing_images(my_autoencoder,"few_faces/Aaron_Patterson/Aaron_Patterson_0001.jpg")

    # loaded_decoder=load_decoder("saved16mars/saved16mars_decoder_fitted.pt")
    # loaded_encoder=load_encoder("saved16mars/saved16mars_encoder_fitted.pt")
    # decoding_images(loaded_decoder,loaded_encoder,"faces/Afton_Smith/Afton_Smith_0001.jpg")
    # my_autoencoder_loaded=loading_autoencoder("autoencoder_fitted_29faces_10epochs.pt")
    # comparing_images(my_autoencoder_loaded,"faces/Afton_Smith/Afton_Smith_0001.jpg")

    # my_autoencoder_loaded=loading_autoencoder("saved16mars/saved16mars_autoencoder.pt")
    # comparing_images(my_autoencoder_loaded,"few_faces/Adam_Ant/Adam_Ant_0001.jpg")
    # comparing_images(my_autoencoder_loaded,"few_faces/Aaron_Patterson/Aaron_Patterson_0001.jpg")
    # comparing_images(my_autoencoder_loaded,"faces/Afton_Smith/Afton_Smith_0001.jpg")
