
from autoencoder import *
from geneticAlgo import *

if __name__ == "__main__":

    loaded_decoder=load_decoder("models/decoder_18_03_15epochs_256batchsize.pt")
    loaded_encoder=load_encoder("models/encoder_18_03_15epochs_256batchsize.pt")
    encoded_image1 = encoding_Image_to_Vector('faces/Afton_Smith/Afton_Smith_0001.jpg',loaded_encoder)
    encoded_image2 = encoding_Image_to_Vector("faces/Azra_Akin/Azra_Akin_0001.jpg",loaded_encoder)
    encoded_image3 = encoding_Image_to_Vector("faces/Aaron_Patterson/Aaron_Patterson_0001.jpg",loaded_encoder)
    # decoding_images(loaded_decoder,loaded_encoder,"faces/Afton_Smith/Afton_Smith_0001.jpg")
    # decoding_images(loaded_decoder,loaded_encoder,"faces/Azra_Akin/Azra_Akin_0001.jpg")
    # decoding_images(loaded_decoder,loaded_encoder,"faces/Aaron_Patterson/Aaron_Patterson_0001.jpg")

    vect_select=[encoded_image1,encoded_image2,encoded_image3]

    fig = plt.figure(figsize=(100,100))
    for i, vector in enumerate(vect_select):
        ax = fig.add_subplot(1,3,i+1)
        decoded_pil=decoding_Vector_to_Image(vector,loaded_decoder)
        ax.imshow(decoded_pil)
        decoded_pil.save(str(i)+".png", format="png")
    plt.show()


    new_vectors=allNewvectors(vect_select,1)
    print(len(new_vectors[0]))
    #mutation only
    fig = plt.figure(figsize=(100,100))
    for i, vector in enumerate(new_vectors):
        ax = fig.add_subplot(1,10,i+1)
        decoded_pil=decoding_Vector_to_Image(vector,loaded_decoder)
        ax.imshow(decoded_pil)
        decoded_pil.save(str(i)+".png", format="png")
    plt.show()

    # #crossing over
    # fig = plt.figure(figsize=(100,100))
    # mut2 = multi_point_crossover(list_mutated_vector)
    # for i, vector in enumerate(mut2):
    #     ax = fig.add_subplot(1,5,i+1)
    #     decoded_pil=decoding_Vector(vector,loaded_decoder)
    #     ax.imshow(decoded_pil)
    #     decoded_pil.save("0"+str(i)+".png", format="png")
    # plt.show()
