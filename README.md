# FBI

## Description

This application was conceived to help a victim or a witness to identify a guilty (or not) person. This project used a face database (FFHQ, Flickr-Faces-HQ) to offer possible supects to the witness. They can then choose the closer face at each round. The application also offers images that are created from scratch thanks to mutation or crossing over of other choosen images. 
The objective is to converge to a single robot portrait that should corrrespond to the suspect that the victim/witness saw.

This package contains 4 modules:

- autoencoder.py, that can convert images to vectors of length 64, and can also decode them.
- geneticAlgo.py, that generates mutated and crossed over vector, that can then be transformed to images.
- application.py, that is the graphical user interface.
- main.py, that launches the application.

This package also contains directories and data that allow our package to run:
- temp: the images selected by the eye-witness/victim at each round, they are temporary and renewed each time they click on "Sélection terminée".
- selected: the history of all images selected in every round.
- images : the images displayed in the current round.
- models: the autoencoder, encoder and decoder created and trained thanks to autoencoder.py.
- docs: the documentation.
- means_of_all_encoded_vector_per_position.txt : a text document containing the mean for each of the 64 positions of an encoded vector, it was generated using a 1000 images of the FFHQ database.
- stds_of_all_encoded_vector_per_position.txt : a text document containing the standard errors for each of the 64 positions of an encoded vector, it was generated using a 1000 images of the FFHQ database.

### Application

The GUI of our project. The victim chooses a selection of images that fits the best the suspect. The window shows faces that are mutated but also faces that are from our DB. You can find in the application a tutorial on how it works.

### Autoencoder

Algorithm used to encode and decode the images. It was used to train an autoencoder model on the FFHQ database. 

The commented parts of the code are of no use to run the applictaion, but were used to create our model.

The model used in this version was trained on 70 000 images (split in training and validation), with 18 epochs and batches of size 16.

### GeneticAlgo

Algorithm that computes mutations and crossing-overs of the encoded images.
Based on a small set (1-5) of received encoded images (arrays of a certain length), the algorithm generates new vectors by performing modifications on the arrays (mutations), and mixing the arrays together (crossing over). The computations pf the algorithm depends on the number of encoded images that are passed as an input.
This algorithm requires the text files "stds_of_all_encoded_vector_per_position.txt" and "means_of_all_encoded_vector_per_position.txt" which contains standard deviations and means of the encoded images for each positions.

## Installation

`python3 -m venv env1`
`source env1/bin/activate` 
`python3 -m pip install --index-url https://test.pypi.org/simple/ --no-deps fbiprojet-group2insa`
`pip install fbi-group2`
`cp -r env1/lib/<python-vers>/site-packages/fbi-group2 path`
`python main.py`

## Authors
Zoé BAPT, Capucine BREANT, Elea PAULIAT, Mathias ROUMANE, Deborah SCHULE

## License
MIT License
