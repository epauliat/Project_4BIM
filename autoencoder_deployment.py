#################
#    IMPORTS    #
#################

import os
import sys
# import time
import torch

import numpy as np
import torch.nn as nn

from PIL import Image
from math import sqrt
from statistics import mean
from matplotlib import pyplot as plt
from matplotlib import image as mpimg
from torchvision import transforms, datasets
# from sklearn.model_selection import train_test_split


###################
#  DATA & IMAGES  #
###################

# def Data_import(dataset_path, batchsize):
#     """Function that imports all the images in a given directory as a DataLoader
#         Args:
#             dataset_path (str): directory path to images
#             batchsize (int): size of batches created by the DataLoader
#         Returns:
#             DataLoader: object containing all the imported images in batches
#     """
#     dataset = datasets.ImageFolder(root=dataset_path, transform=transforms.Compose([transforms.ToTensor(),
#
#                                                                                     transforms.Resize((64,64))])) #transforms.CenterCrop(200),
#
#     X_train, X_validation = train_test_split(dataset,test_size=0.2, random_state=1) # add shuffle option = True ?
#
#     train_dataloader = torch.utils.data.DataLoader(X_train, batch_size = batchsize, shuffle = True)
#     print("Training data acquired")
#     print(f"Training dataloader contains : {len(train_dataloader)} batchs each containing {batchsize} images")
#
#     valid_dataloader = torch.utils.data.DataLoader(X_validation, batch_size = batchsize, shuffle = False)
#     print("Validation data acquired")
#     print(f"Training dataloader contains : {len(valid_dataloader)} batchs each containing {batchsize} images")
#
#
#     return train_dataloader,valid_dataloader

def Image_Conversion_to_tensor(path):
    """Function that converts an imported image in PIL format to a Tensor
        Args:
            path (str): image path from a given directory
        Returns:
            tensor: converted image as a tensor
    """
    image_pil=Image.open(path)
    image_pil_resized=image_pil.resize((64,64))
    transformation=transforms.ToTensor()
    image_tensor=transformation(image_pil_resized)
    return image_tensor


#######################
#  MODELS & TRAINING  #
#      METHODS        #
#######################

class Encoder(nn.Module):
    def __init__(self):
        """Encoder constructor for input tensor of size 3x64x64
            Args:
                None
            Returns:
                None
        """
        super().__init__()

        self.encoder_Conv2d_ReLU_1 = nn.Sequential(
            nn.Conv2d(3, 4, 3, stride=2, padding=1), # stride = 2, padding ='same'
            nn.ReLU()
        )
        self.encoder_Conv2d_ReLU_2 = nn.Sequential(
            nn.Conv2d(4, 8, 3, stride=2, padding=1),
            nn.ReLU()
        )
        self.encoder_MaxPool2d = nn.MaxPool2d(3)
        self.encoder_Flatten = nn.Flatten()
        self.encoder_Linear = nn.Linear(200,64) #16*4*4
        self.BatchNormalization = nn.BatchNorm2d(8)

    def forward(self, input):
        """Function that encodes an image tensor
            Args:
                input (tensor): the input image tensor of size 3x64x64
            Returns:
                tensor: encoded image tensor of size 1x64
        """
        # print(input.size())
        encoded = self.encoder_Conv2d_ReLU_1(input)
        # print(encoded.size())
        encoded = self.encoder_Conv2d_ReLU_2(encoded)
        # print(encoded.size())
        encoded = self.encoder_MaxPool2d(encoded)
        # print(encoded.size())
        encoded = self.BatchNormalization(encoded)
        # print(encoded.size())
        encoded = self.encoder_Flatten(encoded)
        # print("after flatten",encoded.size())
        encoded = self.encoder_Linear(encoded)
        # print(encoded.size())

        return encoded

class Decoder(nn.Module):
    def __init__(self):
        """Decoder constructor for input encoded tensor of size 1x64
            Args:
                None
            Returns:
                None
        """
        super().__init__()
        self.BatchNormalization = nn.BatchNorm2d(8)

        self.decoder_Linear = nn.Linear(64,8*60*60)
        self.decoder_Unflatten = nn.Unflatten(1,[8,60,60])
        self.decoder_ReLu_ConvTranspose2d_2 = nn.Sequential(
            nn.ConvTranspose2d(8, 4, 3, stride=1, padding=0, output_padding=0),
            nn.ReLU()
        )
        self.decoder_ReLu_ConvTranspose2d_1 = nn.Sequential(
            nn.ConvTranspose2d(4, 3, 3, stride=1, padding=0, output_padding=0),
            nn.Sigmoid()
        )

    def forward(self, input):
        """Function that decodes an encoded image tensor
            # Args:
                input (tensor): encoded image tensor of size 1x64
            Returns:
                decoded: the image tensor after being decoded of size 3x64x64
        """
        decoded = self.decoder_Linear(input)
        # print(decoded.size())
        decoded = self.decoder_Unflatten(decoded)
        # print("after unflatten",decoded.size())
        decoded = self.decoder_ReLu_ConvTranspose2d_2(decoded)
        # print(decoded.size())
        decoded = self.decoder_ReLu_ConvTranspose2d_1(decoded)
        # print(decoded.size())
        return decoded

class Autoencoder(nn.Module):
    def __init__(self):
        """Autoencoder constructor that encodes and decodes an input image of
        size 3x64x64 to a tensor of size 1x64
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
                input (tensor): image tensor of size 3x64x64
            Returns:
                tensor: the image tensor after being decoded of size 3x64x64
        """
        encoded = self.encoder(input)
        decoded = self.decoder(encoded)
        return decoded

# def train(nb_epoch, train_data, valid_data, autoencoder):
#     """Function that trains an autoencoder on a given dataset
#         Args:
#             nb_epoch (int): number of epochs to train the model with
#             train_data (DataLoader): imported images in batches
#             data_data (DataLoader): imported images for validation in batches
#             autoencoder (Autoencoder): autoencoder model to train
#         Returns:
#             list: list of losses per epoch
#             list: list of losses per epoch for validation set
#     """
#     # Loss and optimizer definition
#     criterion = nn.MSELoss()
#     optimizer = torch.optim.Adam(autoencoder.parameters(), lr=1e-3, weight_decay=1e-5)
#
#     # Output lists
#     train_losses = []
#     valid_losses = []
#
#     i=0
#     for epoch in range(1,nb_epoch+1):
#         tic = time.time()
#
#         nb_train_batch = len(train_data)
#         nb_valid_batch = len(valid_data)
#
#         # --- Training ---
#         train_loss = 0
#         print("Training")
#         for (index,batch) in enumerate(train_data):
#             print("batch ", index, "/",nb_train_batch, end=', ')
#             l = 0
#             for j in range(len(batch[0])):
#                 img=batch[0][j]
#                 batchsize=len(batch[0])
#                 img = img.reshape(1, 3, 64, 64)
#                 recon=autoencoder(img)
#                 l += criterion(recon, img)
#
#             l.backward()
#             optimizer.step()
#             optimizer.zero_grad()
#             train_loss += float(l)
#
#         train_losses.append(train_loss/nb_train_batch)
#
#         # --- Validation ---
#         valid_loss = 0
#         with torch.no_grad():
#             print("Validation")
#             for (index,batch) in enumerate(valid_data):
#                 print("batch ", index, "/",nb_valid_batch, end=', ')
#                 l = 0
#                 for j in range(len(batch[0])):
#                     img=batch[0][j]
#                     batchsize=len(batch[0])
#                     img = img.reshape(1, 3, 64, 64)
#                     recon=autoencoder(img)
#                     l += float(criterion(recon, img))
#
#                 valid_loss += l
#
#             valid_losses.append(valid_loss/nb_valid_batch)
#
#         toc = time.time()
#         print(f"Epoch {epoch}/{nb_epoch}: training loss = {train_losses[-1]:.5f}, val loss={valid_losses[-1] :.5f}, Training time : {toc-tic:.2f} s")
#
#     return train_losses,valid_losses

# def plot_Losses_Curve(losses_curve_list,losses_curve_valid_list):
#     """Function that plots losses per epoch
#         Args:
#             losses_curve_list (list): loss per epoch
#             losses_curve_valid_list (list): loss per epoch for validation set
#         Returns:
#             None
#     """
#     fig, ax1 = plt.subplots()
#     ax1.set_xlabel("epoch")
#     ax1.set_ylabel("loss")
#     ax1.plot(losses_curve_list, "r")
#     ax1.plot(losses_curve_valid_list, "b")
#     plt.xticks(np.arange(len(losses_curve_list)), np.arange(1, len(losses_curve_list)+1))
#     fig.tight_layout()  # otherwise the right y-label is slightly clipped
#     ax1.set_title("Loss per epoch")
#     plt.show()


######################
#       MODEL        #
#  SAVING & LOADING  #
######################

# def save(model, path):
#     """Function that saves an Autoencoder model to a file
#         Args:
#             model (Autoencoder): the model to save
#         Returns:
#             None
#     """
#     torch.save(model.state_dict(), path)
#
# def save_decoder(model, path):
#     """Function that saves a Decoder model to a file
#         Args:
#             model (Autoencoder): model having a Decoder model as attribute
#         Returns:
#             None
#     """
#     torch.save(model.decoder.state_dict(), path)
#
# def save_encoder(model, path):
#     """Function that saves an Encoder model to a file
#         Args:
#             model (Autoencoder): model having an Encoder model as attribute
#         Returns:
#             None
#     """
#     torch.save(model.encoder.state_dict(), path)

def load_autoencoder(path):
    """Function that loads an Autoencoder model from a given file
        Args:
            path (str): the path of the file in which the Autoencoder is saved
        Returns:
            Autoencoder: the loaded Autoencoder
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
            Decoder: the loaded Decoder
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
            Encoder: the loaded Encoder
    """
    encoder=Encoder()
    encoder.load_state_dict(torch.load(path))
    encoder.eval()
    return encoder


######################
#  ALL IN ONE MODEL  #
#       METHOD       #
######################
#
# def training_Saving_Autoencoder(nb_epoch, batchsize, dataset_path, saving_path, saving_path_decoder,saving_path_encoder):
#     """Function that creates an autoencoder, trains it, and saves it to a file
#         Args:
#             nb_epoch (int): number of epoch to train the model with
#             batchsize (int): batch size of data
#             saving_path (str): path to the file in which the autoencoder is saved
#             dataset_path (str): path to a directory containing the data as images
#             saving_path_decoder (str): path to the file in which the decoder is saved
#             saving_path_encoder (str): path to the file in which the encoder is saved
#         Returns:
#             Autoencoder: the created and trained autoencoder
#     """
#     print("Importing data ...")
#     train_data, valid_data = Data_import(dataset_path, batchsize)
#
#     my_autoencoder=Autoencoder()
#
#     print("Training model ...")
#     train_losses, valid_losses = train(nb_epoch, train_data, valid_data, my_autoencoder)
#
#     plot_Losses_Curve(train_losses,valid_losses)
#
#     save(my_autoencoder, saving_path)
#     save_decoder(my_autoencoder, saving_path_decoder)
#     save_encoder(my_autoencoder, saving_path_encoder)
#
#     return my_autoencoder


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
    X=image_tensor.reshape(1,3,64,64)
    decoded_tensor=autoencoder.forward(X)
    decoded_pil=transforms.functional.to_pil_image(decoded_tensor.reshape(3,64,64))

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
    X=image_tensor.reshape(1,3,64,64)
    encoded_vector=encoder.forward(X)
    # print(encoded_vector)
    decoded_tensor=decoder.forward(encoded_vector)
    decoded_pil=transforms.functional.to_pil_image(decoded_tensor.reshape(3,64,64))

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
            Numpy Array: image encoded stored in an intermediate vector
    """
    image_tensor= Image_Conversion_to_tensor(path)
    X=image_tensor.reshape(1,3,64,64)
    encoded_vector=encoder.forward(X)
    encoded_vector = encoded_vector.detach().numpy()
    return encoded_vector

def compute_Std_Per_Position_In_Encoded_Vectors(encoder, file_path):
    """Function that computes and saves to txt files the mean and std of the
     values of all_encoded vector from all images of our dataset per position.
        Args:
            encoder(Encoder): trained encoder
            file_path (str): path to the images directory
        Returns:
            None
    """
    print("-- Std Computation --")

    list_encoded_vectors = []

    print("Going through all the files ...")

    files = os.listdir(file_path)
    for name in files:
        picture = os.listdir(file_path+'/'+name)
        for p in picture:
            path = file_path+'/'+name+"/"+p
            list_encoded_vectors.append(encoding_Image_to_Vector(path,encoder))

    print("All vectors encoded, computing std per position ... ")

    means = []
    stds = []
    for pos in range(len(list_encoded_vectors[0])):
        all_values = []
        for vec in list_encoded_vectors:
            all_values.append(vec[pos])

        means.append(sum(all_values)/len(all_values))
        stds.append(np.std(all_values,axis=0))

    np.savetxt("means_of_all_encoded_vector_per_position.txt", means)
    print("Stds saved in means_of_all_encoded_vector_per_position.txt file")
    np.savetxt("stds_of_all_encoded_vector_per_position.txt", stds)
    print("Stds saved in stds_of_all_encoded_vector_per_position.txt file")

def decoding_Vector_to_Image(vector, decoder):
    """Function that decodes a vector to an image
        Args:
            vector(Numpy Array): input encoded image as a vector
            decoder (Decoder): decoder used
        Returns:
            PIL Image: decoded image in PIL format
    """
    decoded_tensor=decoder.forward(torch.tensor(vector))
    decoded_pil=transforms.functional.to_pil_image(decoded_tensor.reshape(3,64,64))
    return decoded_pil


if __name__ == "__main__":

    # TRAINING
    print("main")
    # epoch = 1
    # batch_size = 16
    # my_autoencoder = training_Saving_Autoencoder(epoch,
    #                                             batch_size,
    #                                             'faces',
    #                                             "models/Autoencoder"+str(epoch)+"epochs_"+str(batch_size)+"batchsize.pt",
    #                                             "models/Decoder"+str(epoch)+"epochs_"+str(batch_size)+"batchsize.pt",
    #                                             "models/Encoder"+str(epoch)+"epochs_"+str(batch_size)+"batchsize.pt")
    # comparing_images(my_autoencoder,"faces/1.png")
    # comparing_images(my_autoencoder,"faces/2.png")
    # comparing_images(my_autoencoder,"faces/3.png")


    # LOADING ENCODER & DECODER SEPARATELY

    loaded_encoder=load_encoder("models/encoder.pt")
    loaded_decoder=load_decoder("models/decoder.pt")

    decoding_images(loaded_encoder,loaded_decoder,"faces/1.png")
    # decoding_images(loaded_encoder,loaded_decoder,"faces/2.png")
    # decoding_images(loaded_encoder,loaded_decoder,"faces/3.png")


    # LOADING AUTOENCODER

    # my_autoencoder_loaded=load_autoencoder("models/autoencoder.pt")
    # comparing_images(my_autoencoder,"faces/1.png")
    # comparing_images(my_autoencoder,"faces/2.png")
    # comparing_images(my_autoencoder,"faces/3.png")


    # COMPUTING STD OF ENCODED VECTORS

    # compute_Std_Per_Position_In_Encoded_Vectors(loaded_encoder, 'faces')
