#################
#    IMPORTS    #
#################

import os
import sys
import time
import torch

import numpy as np
import torch.nn as nn

from PIL import Image
from torchsummary import summary
from matplotlib import pyplot as plt
from matplotlib import image as mpimg
from torchvision import transforms, datasets
from keras_preprocessing.image import img_to_array

###################
#  DATA & IMAGES  #
###################

def Data_import(path, batchsize):
    """Function that imports all the images in a given directory as a DataLoader
    Args:
        path (str): directory path to images,
        batchsize (int): size of batches created by the DataLoader
    Returns:
        loader (DataLoader): object containing all the imported images in batches
    """
    dataset = datasets.ImageFolder(root=path, transform=transforms.Compose([transforms.Resize((128,128)),transforms.ToTensor()]))
    loader = torch.utils.data.DataLoader(dataset, batch_size = batchsize, shuffle = True)
    return loader

def Image_Conversion_to_tensor(path):
    """Function that converts an imported image in PIL format to a Tensor
    Args:
        path (str): image path from a given directory
    Returns:
        image_tensor (tensor): converted image as a tensor
    """
    image_pil=Image.open(path)
    image_pil_resized=image_pil.resize((128,128))
    transformation=transforms.ToTensor()
    image_tensor=transformation(image_pil_resized)
    return image_tensor


#######################
#  MODELS & TRAINING  #
#      METHODS        #
#######################

class Encoder(nn.Module):
    def __init__(self):
        """Encoder constructor for input tensor of size 3x128x128
        Args:
            None
        Returns:
            None
        """
        super().__init__()

        self.encoder_Conv2d_ReLU_1 = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=1, padding=0),
            nn.ReLU()
        )
        self.encoder_Conv2d_ReLU_2 = nn.Sequential(
            nn.Conv2d(16, 16, 3, stride=1, padding=0),
            nn.ReLU()
        )
        self.encoder_MaxPool2d = nn.MaxPool2d(3)
        self.encoder_Flatten = nn.Flatten()
        self.encoder_Linear = nn.Linear(144,128)
        self.BatchNormalization = nn.BatchNorm2d(16)

    def forward(self, input):
        """Function that encodes an image tensor
        Args:
            input (tensor): the input image tensor of size 3x128x128
        Returns:
            encoded_Linear: encoded image tensor of size 1x128
        """

        encoded = self.encoder_Conv2d_ReLU_1(input)
        encoded = self.encoder_MaxPool2d(encoded)
        # print(encoded.size())
        encoded = self.encoder_Conv2d_ReLU_2(encoded)
        encoded = self.encoder_MaxPool2d(encoded)
        # print(encoded.size())
        encoded = self.BatchNormalization(encoded)
        # print(encoded.size())
        encoded = self.encoder_Conv2d_ReLU_2(encoded)
        encoded = self.encoder_MaxPool2d(encoded)
        # print(encoded.size())
        encoded = self.encoder_Flatten(encoded)
        # print("after flatten",encoded.size())
        encoded = self.encoder_Linear(encoded)
        # print(encoded.size())

        return encoded

class Decoder(nn.Module):

    def __init__(self):
        """Decoder constructor for input encoded tensor of size 1x128
        Args:
            None
        Returns:
            None
        """
        super().__init__()
        self.BatchNormalization = nn.BatchNorm2d(16)
        self.decoder_Linear = nn.Linear(128,238144)
        self.decoder_Unflatten = nn.Unflatten(1,[16,122,122])
        self.decoder_ReLu_ConvTranspose2d_2 = nn.Sequential(
            nn.ConvTranspose2d(16, 16, 3, stride=1, padding=0, output_padding=0),
            nn.ReLU()
        )
        self.decoder_ReLu_ConvTranspose2d_1 = nn.Sequential(
            nn.ConvTranspose2d(16, 3, 3, stride=1, padding=0, output_padding=0),
            nn.Sigmoid()
        )

    def forward(self, input):
        """Function that decodes an encoded image tensor
        Args:
            input (tensor): encoded image tensor of size 1x128
        Returns:
            decoded: the image tensor after being decoded of size 3x128x128
        """
        decoded = self.decoder_Linear(input)
        # print(decoded.size())
        decoded = self.decoder_Unflatten(decoded)
        # print("after unflatten",decoded.size())
        decoded = self.decoder_ReLu_ConvTranspose2d_2(decoded)
        # print(decoded.size())
        decoded = self.BatchNormalization(decoded)
        # print(decoded.size())
        decoded = self.decoder_ReLu_ConvTranspose2d_2(decoded)
        # print(decoded.size())
        decoded = self.decoder_ReLu_ConvTranspose2d_1(decoded)
        # print(decoded.size())
        return decoded

class Autoencoder(nn.Module):
    def __init__(self):
        """Autoencoder constructor that encodes and decodes an input image of
        size 3x128x128 to a tensor of size 1x128
        Args:
            None
        Returns:
            None
        """
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, input):
        """Function that encodes an image, then decodes it
        Args:
            input (tensor): image tensor of size 3x128x128
        Returns:
            decoded: the image tensor after being decoded of size 3x128x128
        """
        encoded = self.encoder(input)
        decoded = self.decoder(encoded)
        return decoded

def train(nb_epoch, data, autoencoder):
    """Function that trains an autoencoder on a given dataset
    Args:
        nb_epoch (int): number of epochs to train the model with
        data (DataLoader): imported images in batches
        autoencoder (Autoencoder): autoencoder model to train
    Returns:
        losses_curve (list): list of losses per epoch
    """
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=1e-3, weight_decay=1e-5)
    outputs = []
    losses_curve = []
    i=0
    for epoch in range(nb_epoch):
        tic = time.time()

        for (index, batch) in enumerate(data):
            loss=0
            loss_curve=0
            print("batch number "+str(index), end="  :")

            for j in range(len(batch[0])):
                img=batch[0][j]
                batchsize=len(batch[0])
                img = img.reshape(1, 3, 128, 128)
                recon=autoencoder(img)

                if(i%10==0):
                    print(str(i), end=",")

                sys.stdout.flush()
                i+=1
                loss += criterion(recon, img)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            loss_curve += float(loss/len(batch[0]))
            print("")

        i=0
        losses_curve.append(loss_curve)
        toc = time.time()
        print(f'__________Epoch:{epoch+1}, Loss:{loss.item():.4f}, Training time : {toc-tic:.2f} s')
        outputs.append((epoch, img, recon))

    return losses_curve

def plot_Losses_Curve(losses_curve_list):
    """Function that plots losses per epoch
    Args:
        losses_curve_list (list): loss per epoch
    Returns:
        None
    """
    fig, ax1 = plt.subplots()
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("loss")
    ax1.plot(losses_curve_list, "r")
    plt.xticks(np.arange(len(losses_curve_list)), np.arange(1, len(losses_curve_list)+1))
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    ax1.set_title("Loss per epoch")
    plt.show()


######################
#       MODEL        #
#  SAVING & LOADING  #
######################

def save(model, path):
    """Function that saves an Autoencoder model to a file
    Args:
        model (Autoencoder): the model to save
    Returns:
        None
    """
    torch.save(model.state_dict(), path)

def save_decoder(model, path):
    """Function that saves a Decoder model to a file
    Args:
        model (Autoencoder): model having a Decoder model as attribute
    Returns:
        None
    """
    torch.save(model.decoder.state_dict(), path)

def save_encoder(model, path):
    """Function that saves an Encoder model to a file
    Args:
        model (Autoencoder): model having an Encoder model as attribute
    Returns:
        None
    """
    torch.save(model.encoder.state_dict(), path)

def load_autoencoder(path):
    """Function that loads an Autoencoder model from a given file
    Args:
        path (str): the path of the file in which the Autoencoder is saved
    Returns:
        autoencoder (Autoencoder): the loaded Autoencoder
    """
    autoencoder = Autoencoder()
    autoencoder.load_state_dict(torch.load(path))
    autoencoder.eval()
    return autoencoder

def load_decoder(path):
    """Function that loads a Decoder from a given file
    Args:
        path (str): the path of the file in which the Decoder is saved
    Returns:
        decoder (Decoder): the loaded Decoder
    """
    decoder=Decoder()
    decoder.load_state_dict(torch.load(path))
    decoder.eval()
    return decoder

def load_encoder(path):
    """Function that loads a Encoder from a given file
    Args:
        path (str): the path of the file in which the Encoder is saved
    Returns:
        encoder (Encoder): the loaded Encoder
    """
    encoder=Encoder()
    encoder.load_state_dict(torch.load(path))
    encoder.eval()
    return encoder

######################
#  ALL IN ONE MODEL  #
#       METHOD       #
######################

def training_Saving_Autoencoder(nb_epoch, batchsize, saving_path, data_path,saving_path_decoder,saving_path_encoder):
    """Function that creates an autoencoder, trains it, and saves it to a file
    Args:
        nb_epoch (int): number of epoch to train the model with
        batchsize (int): batch size of data
        saving_path (str): path to the file in which the autoencoder is saved
        data_path (str): path to a directory containing the data as images
        saving_path_decoder (str): path to the file in which the decoder is saved
        saving_path_encoder (str): path to the file in which the encoder is saved
    Returns:
        my_autoencoder (Autoencoder): the created and trained autoencoder
    """
    data=Data_import(data_path, batchsize)
    print("Data acquired")
    my_autoencoder=Autoencoder()
    losses = train(nb_epoch, data, my_autoencoder)
    plot_Losses_Curve(losses)
    save(my_autoencoder, saving_path)
    save_decoder(my_autoencoder, saving_path_decoder)
    save_encoder(my_autoencoder, saving_path_encoder)
    return my_autoencoder


#########################
#  MODEL VISUALISATION  #
#########################


def comparing_images(autoencoder, path_to_image):
    """Function that shows 2 images: the original one and the recomposed one
    /!\ from an Autoencoder model /!\
    Args:
        autoencoder (Autoencoder): the autoencoder used
        path_to_image (str): the path we want to load the image from
    Returns:
        None
    """
    image_tensor= Image_Conversion_to_tensor(path_to_image)
    X=image_tensor.reshape(1,3,128,128)
    decoded_tensor=autoencoder.forward(X)
    decoded_pil=transforms.functional.to_pil_image(decoded_tensor.reshape(3,128,128))

    f,axs=plt.subplots(1,2,figsize=(5,5))
    axs[0].imshow(decoded_pil)
    axs[1].imshow(mpimg.imread(path_to_image))
    plt.show()

def decoding_images(encoder, decoder, path_to_image):
    """Function that shows 2 images: the original one and the recomposed one
    /!\ from an encoder and a decoder /!\
    Args:
        encoder (Encoder): the encoder used
        decoder (Decoder): the decoder used
        path_to_image (str): the path we want to load the image from
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


########################
#  COMPATIBILITY WITH  #
#  GENETIC ALGO CODE   #
########################

def encoding_Image_to_Vector(path, encoder):
    """Function that encodes an image to a vector
    Args:
        path (str): path to an image
        encoder (Encoder): encoder used
    Returns:
        encoded_vector (Numpy Array): image encoded stored in an intermediate vector
    """
    image_tensor= Image_Conversion_to_tensor(path)
    X=image_tensor.reshape(1,3,128,128)
    encoded_vector=encoder.forward(X)
    encoded_vector = encoded_vector.detach().numpy()
    return encoded_vector

def decoding_Vector_to_Image(vector, decoder):
    """Function that decodes a vector to an image
    Args:
        vector(Numpy Array): input encoded image as a vector
        decoder (Decoder): decoder used
    Returns:
        decoded_pil (PIL Image): decoded image in PIL format
    """
    decoded_tensor=decoder.forward(torch.tensor(vector)) #, indices
    decoded_pil=transforms.functional.to_pil_image(decoded_tensor.reshape(3,128,128))
    return decoded_pil


if __name__ == "__main__":

    # TRAINING

    epoch = 2
    batch_size = 20
    my_autoencoder=training_Saving_Autoencoder(epoch, batch_size, "models/autoencoder_18_03.pt",'few_faces',"models/decoder_18_03.pt","models/encoder_18_03.pt")
    comparing_images(my_autoencoder,"few_faces/Aaron_Patterson/Aaron_Patterson_0001.jpg")
    comparing_images(my_autoencoder,"faces/Adam_Ant/Adam_Ant_0001.jpg")
    comparing_images(my_autoencoder,"faces/Afton_Smith/Afton_Smith_0001.jpg")

    # LOADING ENCODER & DECODER SEPARATELY

    # loaded_decoder=load_decoder("saved16mars/saved16mars_decoder_fitted.pt")
    # loaded_encoder=load_encoder("saved16mars/saved16mars_encoder_fitted.pt")
    # decoding_images(loaded_decoder,loaded_encoder,"faces/Afton_Smith/Afton_Smith_0001.jpg")
    # my_autoencoder_loaded=load("autoencoder_fitted_29faces_10epochs.pt")
    # comparing_images(my_autoencoder_loaded,"faces/Afton_Smith/Afton_Smith_0001.jpg")

    # LOADING AUTOENCODER
    #
    # my_autoencoder_loaded=load_autoencoder("models/autoencoder_18_03.pt")
    # comparing_images(my_autoencoder_loaded,"few_faces/Adam_Ant/Adam_Ant_0001.jpg")
    # comparing_images(my_autoencoder_loaded,"few_faces/Aaron_Patterson/Aaron_Patterson_0001.jpg")
    # comparing_images(my_autoencoder_loaded,"faces/Afton_Smith/Afton_Smith_0001.jpg")
