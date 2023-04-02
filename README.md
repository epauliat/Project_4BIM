# FBI

## Description

This application was conceived to help a victim or a witness to identify a guilty (or not) person. This project used a face database (FFHQ, Flickr-Faces-HQ) to offer possible supects to the witness. They can then choose the closer face at each round. The application also offers images that are created from scratch thanks to mutation or crossing over of other choosen images. 
The objective is to converge to a single robot portrait that should corrrespond to the suspect that the victim/witness saw.

This package contains 3 modules:

- autoencoder.py, that can convert images to vectors of length 64, and can also decode them.
- geneticAlgo.py, that generates mutated and crossed over vector, that can then be transformed to images.
- application.py, that is the graphical user interface.

This package also contains tutorials, directory and data that allow our package to run.

### Application

The GUI of our project. The victim chooses a selection of images that fit the best the suspect. The window shows faces that are mutated but also faces that are from our DB. You can find in the application a tutorial on how it works.

### Autoencoder

Algorithm used to encode and decode the images. It was used to train an autoencoder model on the FFHQ database. 

The commented parts of the code are of no use to run the applictaion, but were used to create our model.

The model used in this version was trained on 70 000 images (split in training and validation), with 18 epochs and batches of size 16.

### GeneticAlgo

Algorithm that computes mutations and crossing-overs of the encoded images.
Based on a small set (1-5) of received encoded images (arrays of a certain length), the algorithm generates new vectors by performing modifications on the arrays (mutations), and mixing the arrays together (crossing over). The computations pf the algorithm depends on the number of encoded images that are passed as an input.
This algorithm requires the text files "stds_of_all_encoded_vector_per_position.txt" and "means_of_all_encoded_vector_per_position.txt" which contains standard deviations and means of the encoded images for each positions.

## Installation

`pip install fbi`

## Getting Started

...here the tutorial for installation

## Authors
Zoé BAPT, Capucine BREANT, Elea PAULIAT, Mathias ROUMANE, Deborah SCHULE

## License
MIT License
