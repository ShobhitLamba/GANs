# Implementation of a Deep Convolutional Generative Adversarial Network running over MNIST dataset
# Author: Shobhit Lamba
# e-mail: slamba4@uic.edu

# Importing the libraries
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Reshape
from keras.layers import Conv2D, Conv2DTranspose, UpSampling2D
from keras.layers import LeakyReLU, Dropout
from keras.layers import BatchNormalization
from keras.optimizers import RMSprop
from keras.datasets import mnist

batch_size = 128
epochs = 5

# Importing the dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = np.concatenate((x_train, x_test), axis = 0) 
x_train = x_train.astype("float32") / 255

# Defining a Discriminator that tells how real an image fed to it is
def discriminator(self):
    self.D = Sequential()
    
    depth = 64
    dropout = 0.4    
    input_shape = (self.img_rows, self.img_cols, self.channel)
    
    self.D.add(Conv2D(depth * 1, 5, strides = 2, input_shape = input_shape, padding = "same"))
    self.D.add(LeakyReLU(alpha = 0.2))
    self.D.add(Dropout(dropout))
    
    self.D.add(Conv2D(depth*2, 5, strides = 2, padding = "same"))
    self.D.add(LeakyReLU(alpha = 0.2))
    self.D.add(Dropout(dropout))

    self.D.add(Conv2D(depth*4, 5, strides = 2, padding = "same"))
    self.D.add(LeakyReLU(alpha = 0.2))
    self.D.add(Dropout(dropout))

    self.D.add(Conv2D(depth*8, 5, strides = 1, padding = "same"))
    self.D.add(LeakyReLU(alpha = 0.2))
    self.D.add(Dropout(dropout))
    
    # Output: Probability
    self.D.add(Flatten())
    self.D.add(Dense(1, activation = "sigmoid"))
    self.D.summary()
    
    return self.D
    
# Defining a Generator that generates a grayscale image    
def generator(self):
    self.G = Sequential()
    
    depth = 256
    dropout = 0.4
    dim = 7
    
    self.G.add(Dense(dim * dim * depth, input_dim = 100))
    self.G.add(BatchNormalization(momentum = 0.9))
    self.G.add(Activation("relu"))
    self.G.add(Reshape((dim, dim, depth)))
    self.G.add(Dropout(dropout))
    
    self.G.add(UpSampling2D())
    self.G.add(Conv2DTranspose(int(depth / 2), 5, padding = "same"))
    self.G.add(BatchNormalization(momentum = 0.9))
    self.G.add(Activation("relu"))
    self.G.add(UpSampling2D())
    
    self.G.add(Conv2DTranspose(int(depth / 4), 5, padding = "same"))
    self.G.add(BatchNormalization(momentum = 0.9))
    self.G.add(Activation("relu"))
    
    self.G.add(Conv2DTranspose(int(depth / 8), 5, padding = "same"))
    self.G.add(BatchNormalization(momentum = 0.9))
    self.G.add(Activation("relu"))
    
    # Output: A grayscale image
    self.G.add(Conv2DTranspose(1, 5, padding = "same"))
    self.G.add(Activation("sigmoid"))
    self.G.summary()
    
    return self.G

# Creating a discriminator model using the discriminator()
def discriminator_model(self):
    optimizer = RMSprop(lr = 0.0002, decay = 6e-8)
    
    self.DM.add = Sequential()
    self.DM.add(self.discriminator())
    self.DM.compile(loss = "binary_crossentropy", optimizer = optimizer, metrics = ["accuracy"])
    
    return self.DM

# Creating an adversarial model using the generator() and discriminator()
def adversarial_model(self):
    optimizer = RMSprop(lr = 0.0001, decay = 3e-8)
    
    self.AM = Sequential()
    self.AM.add(self.generator())
    self.AM.add(self.discriminator())
    self.AM.compile(loss = "binary_crossentropy", optimizer = optimizer, metrics = ["accuracy"])
        
    return self.AM

class DCGAN(object):
    def __init__(self, img_rows = 28, img_cols = 28, channel = 1):

        self.img_rows = img_rows
        self.img_cols = img_cols
        self.channel = channel
        self.D = None   # discriminator
        self.G = None   # generator
        self.AM = None  # adversarial model
        self.DM = None  # discriminator model
    