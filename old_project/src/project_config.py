import numpy as np
import tensorflow as tf
import idx2numpy
import sys,os
import matplotlib.pyplot as plt
# import Plotting_class
###############


### CONSTANTS ###
LAYERS_NUM = 5
C = 299792458                               # Meter/sec
PIXEL_HEIGHT = PIXEL_WIDTH  = 400E-06       # Meter
PIXEL_NUM_X = PIXEL_NUM_Y = 200
LAYERS_DISTANCE = 0.03                      # Meter
LIGHT_FREQ = 0.4E+12                        # 1/sec
WAVELENGTH = C/LIGHT_FREQ                   # M
VACUUM_n = 1                                # refractive idx
MEDIUM_n = 1.7227                           # medium refractive idx
EXTINCTION_COEFF = 0.0311                   # as used in original paper
ATTENUATION_COEFF = 520.7177E-1             # 1/Meter

FIXED_WAVELENGTH = WAVELENGTH/400E-6
FIXED_PIXEL_HEIGHT = FIXED_PIXEL_WIDTH  = 1 # 1 unit pixel - for nrefocus moduls
FIXED_LAYERS_DISTANCE = 75 # equate to 3 cm

PI = np.pi
EXP = np.exp
##################


### PATHS ###
MNIST_DIGIT_TRAIN_IMGS_PATH = "C:/Users/Avshalom/Desktop/Project/MNIST_DIGITS/DB/train-images" \
                              ".idx3-ubyte"
MNIST_DIGIT_TRAIN_LABELS_PATH = "C:/Users/Avshalom/Desktop/Project/MNIST_DIGITS/DB/train-labels" \
                                ".idx1-ubyte"
MNIST_DIGIT_TEST_IMGS_PATH = "C:/Users/Avshalom/Desktop/Project/MNIST_DIGITS/DB/t10k-images.idx3" \
                             "-ubyte"
MNIST_DIGIT_TEST_LABELS_PATH = "C:/Users/Avshalom/Desktop/Project/MNIST_DIGITS/DB/t10k-labels" \
                               ".idx1-ubyte"
MNIST_DIGIT_TRAIN_LABELS_CONVERTED_PATH = \
    "C:/Users/Avshalom/Desktop/Project/MNIST_DIGITS/DB/TARGET_LABELS_28x28_FULL.idx3-ubyte"
