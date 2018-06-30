# Implementation of a Deep Convolutional Generative Adversarial Network running over MNIST dataset
# Author: Shobhit Lamba
# e-mail: slamba4@uic.edu

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Reshape
from keras.layers import Conv2D, Conv2DTranspose, UpSampling2D
from keras.layers import LeakyReLU, Dropout
from keras.layers import BatchNormalization
from keras.optimizers import RMSprop
from tensorflow.examples.tutorials.mnist import input_data

# Class defining the structure of the DCGAN
class DCGAN(object):
    def __init__(self, img_rows = 28, img_cols = 28, channel = 1):

        self.img_rows = img_rows
        self.img_cols = img_cols
        self.channel = channel
        self.D = None   # discriminator
        self.G = None   # generator
        self.AM = None  # adversarial model
        self.DM = None  # discriminator model

    # Defining a Discriminator that tells how real an image fed to it is
    def discriminator(self):
        if self.D:
            return self.D

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

        # Output: Probability
        self.D.add(Flatten())
        self.D.add(Dense(1, activation = "sigmoid"))
        self.D.summary()

        return self.D

    # Defining a Generator that generates a grayscale image
    def generator(self):
        if self.G:
            return self.G

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

        # Output: A grayscale image
        self.G.add(Conv2DTranspose(1, 5, padding = "same"))
        self.G.add(Activation("sigmoid"))
        self.G.summary()

        return self.G

    # Creating a discriminator model using the discriminator()
    def discriminator_model(self):
        if self.DM:
            return self.DM

        optimizer = RMSprop(lr = 0.0008, clipvalue = 1.0, decay = 6e-8)

        self.DM = Sequential()
        self.DM.add(self.discriminator())
        self.DM.compile(loss = "binary_crossentropy", optimizer = optimizer, metrics = ["accuracy"])

        return self.DM

    # Creating an adversarial model using the generator() and discriminator()
    def adversarial_model(self):
        if self.AM:
            return self.AM

        optimizer = RMSprop(lr = 0.0004, clipvalue = 1.0, decay = 3e-8)

        self.AM = Sequential()
        self.AM.add(self.generator())
        self.AM.add(self.discriminator())
        self.AM.compile(loss = "binary_crossentropy", optimizer = optimizer, metrics = ["accuracy"])

        return self.AM

# Class to train the DCGAN and generating images
class TRAIN_DCGAN(object):
    def __init__(self):
        self.img_rows = 28
        self.img_cols = 28
        self.channel = 1

        self.x_train = input_data.read_data_sets("mnist",\
        	one_hot=True).train.images
        self.x_train = self.x_train.reshape(-1, self.img_rows,\
        	self.img_cols, 1).astype(np.float32)

        self.DCGAN = DCGAN()
        self.discriminator = self.DCGAN.discriminator_model()
        self.adversarial = self.DCGAN.adversarial_model()
        self.generator = self.DCGAN.generator()

    def train(self, train_steps = 10, batch_size = 256, save_interval = 0):
        noise_input = None
        if save_interval > 0:
            noise_input = np.random.uniform(-1.0, 1.0, size = [16, 100])
        for i in range(train_steps):
            images_train = self.x_train[np.random.randint(0,
                self.x_train.shape[0], size = batch_size), :, :, :]
            noise = np.random.uniform(-1.0, 1.0, size = [batch_size, 100])
            images_fake = self.generator.predict(noise)
            x = np.concatenate((images_train, images_fake))
            y = np.ones([2*batch_size, 1])
            y[batch_size:, :] = 0
            d_loss = self.discriminator.train_on_batch(x, y)

            y = np.ones([batch_size, 1])
            noise = np.random.uniform(-1.0, 1.0, size = [batch_size, 100])
            a_loss = self.adversarial.train_on_batch(noise, y)
            log_mesg = "%d: [D loss: %f, acc: %f]" % (i, d_loss[0], d_loss[1])
            log_mesg = "%s  [A loss: %f, acc: %f]" % (log_mesg, a_loss[0], a_loss[1])
            print(log_mesg)
            if save_interval>0:
                if (i+1)%save_interval == 0:
                    self.plot_images(save2file = True, samples = noise_input.shape[0],\
                        noise = noise_input, step = (i+1))

    def plot_images(self, save2file = False, fake = True, samples = 16, noise = None, step = 0):
        filename = "mnist.png"
        if fake:
            if noise is None:
                noise = np.random.uniform(-1.0, 1.0, size = [samples, 100])
            else:
                filename = "mnist_%d.png" % step
            images = self.generator.predict(noise)
        else:
            i = np.random.randint(0, self.x_train.shape[0], samples)
            images = self.x_train[i, :, :, :]

        plt.figure(figsize = (10,10))
        for i in range(images.shape[0]):
            plt.subplot(4, 4, i+1)
            image = images[i, :, :, :]
            image = np.reshape(image, [self.img_rows, self.img_cols])
            plt.imshow(image, cmap = "gray")
            plt.axis("off")
        plt.tight_layout()
        if save2file:
            plt.savefig(filename)
            plt.close("all")
        else:
            plt.show()

if __name__ == '__main__':
    train_dcgan = TRAIN_DCGAN()
    train_dcgan.train(train_steps = 100, batch_size = 256, save_interval = 500)
    train_dcgan.plot_images(fake = True)
    train_dcgan.plot_images(fake = False, save2file = True)
