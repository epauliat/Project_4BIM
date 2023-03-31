from autoencoder import *
from geneticAlgo import *

def mutating(encoded_vectors, probability, std_file_path, show=False):
    """Functions that mutates the encoded images and prints them
        Args:
            encoded_vectors (list): list of encoded vectors to be mutated
            probability (float): probability used for mutations
            std_file_path (str): path to the std txt file
            show (boolean): if True, mutated vectors are shown in a pop-up
        Returns:
            list: list of encoded mutated images

    """
    # LOADING THE STDS PER POSITION FROM TEXT FILE

    std = []
    with open(std_file_path) as f:
        std = f.readlines()
    stds = std[0].split(' ')

    for i in range(len(stds)):
        stds[i]=float(stds[i])

    # MUTATING ENCODED VECTORS

    new_vectors=allNewvectors(vect_select,probability,stds)

    # PLOTTING MUTATED SOLUTIONS

    fig = plt.figure(figsize=(100,100))
    for i, vector in enumerate(new_vectors):
        ax = fig.add_subplot(1,10,i+1)
        decoded_pil=decoding_Vector_to_Image(vector,loaded_decoder)
        ax.imshow(decoded_pil)
        decoded_pil.save(str(i)+".png", format="png")
    if(show==True):
        plt.show()

    return new_vectors




if __name__ == "__main__":

    # LOADING MODELS

    loaded_decoder=load_decoder("models/BIS70000decoder_31_03_12epoch_16batchsize_64size_stride_padding.pt")
    loaded_encoder=load_encoder("models/BIS70000encoder_31_03_12epoch_16batchsize_64size_stride_padding.pt")

    # ENCODING TEST IMAGES

    encoded_image1 = encoding_Image_to_Vector("faces_bis/43000/43002.png",loaded_encoder)
    encoded_image2 = encoding_Image_to_Vector("faces_bis/43000/43011.png",loaded_encoder)
    encoded_image3 = encoding_Image_to_Vector("faces_bis/43000/43020.png",loaded_encoder)

    vect_select=[encoded_image1,encoded_image2,encoded_image3]

    # DECODING IMAGES

    decoding_images(loaded_encoder,loaded_decoder,"faces_bis/43000/43002.png")
    decoding_images(loaded_encoder,loaded_decoder,"faces_bis/43000/43011.png")
    decoding_images(loaded_encoder,loaded_decoder,"faces_bis/43000/43020.png")

    # PLOTTING DECODED VECTORS

    fig = plt.figure(figsize=(100,100))
    for i, vector in enumerate(vect_select):
        ax = fig.add_subplot(1,len(vect_select),i+1)
        decoded_pil=decoding_Vector_to_Image(vector,loaded_decoder)
        ax.imshow(decoded_pil)
        decoded_pil.save(str(i)+".png", format="png")
    plt.show()


    mutating(vect_select,1.5,"stds_of_all_encoded_vector_per_position.txt", show=True)
