
from encoder import *
from geneticAlgo import *

if __name__ == "__main__":

    loaded_decoder=load_decoder("decoder_fitted_test.pt")
    loaded_encoder=load_encoder("encoder_fitted_test.pt")
    encoded_image1 = encodedVectorGenerator('faces/Afton_Smith/Afton_Smith_0001.jpg',loaded_encoder)
    encoded_image2 = encodedVectorGenerator("faces/Azra_Akin/Azra_Akin_0001.jpg",loaded_encoder)
    encoded_image3 = encodedVectorGenerator("faces/Aaron_Patterson/Aaron_Patterson_0001.jpg",loaded_encoder)
    # decoding_images(loaded_decoder,loaded_encoder,"faces/Afton_Smith/Afton_Smith_0001.jpg")
    # decoding_images(loaded_decoder,loaded_encoder,"faces/Azra_Akin/Azra_Akin_0001.jpg")
    # decoding_images(loaded_decoder,loaded_encoder,"faces/Aaron_Patterson/Aaron_Patterson_0001.jpg")

    vect_select=[encoded_image1,encoded_image2,encoded_image3]
    mutants_select=liste_mutants_select(vect_select,0)
    list_mutated_vector=mutants_complets(vect_select,mutants_select,0)

    #mutation only
    fig = plt.figure(figsize=(100,100))
    for i, vector in enumerate(list_mutated_vector):
        ax = fig.add_subplot(1,5,i+1)
        decoded_pil=decoding_Vector(vector,loaded_decoder)
        ax.imshow(decoded_pil)
        decoded_pil.save(str(i)+".png", format="png")
    plt.show()

    #crossing over
    fig = plt.figure(figsize=(100,100))
    mut2 = multi_point_crossover(list_mutated_vector)
    for i, vector in enumerate(mut2):
        ax = fig.add_subplot(1,5,i+1)
        decoded_pil=decoding_Vector(vector,loaded_decoder)
        ax.imshow(decoded_pil)
        decoded_pil.save("0"+str(i)+".png", format="png")
    plt.show()
