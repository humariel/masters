import glob
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import time
import tensorflow as tf
from tensorflow.keras import layers
import util
from math import sqrt
from util import load_isic_dataset
import pickle

# generate points in latent space as input for the generator
def generate_latent_points_class(latent_dim, n_samples, n_class):
    # generate points in the latent space
    z_input = np.random.randn(n_samples, latent_dim)
    # generate labels
    labels = np.asarray([n_class for _ in range(n_samples)])
    return [z_input, labels]


# create and save a plot of generated images
def save_plot(examples, n_examples, filename):
    # plot images
    for i in range(n_examples):
        # define subplot
        plt.subplot(sqrt(n_examples), sqrt(n_examples), 1 + i)
        # turn off axis
        plt.axis('off')
        # plot raw pixel data
        plt.imshow(examples[i, :, :, 0], cmap="gray")
    plt.savefig(filename)


class ACGAN():
    def __init__(self, epochs=50, batch_sz=256, latent_dim=100, n_classes=10):
        self.epochs = epochs
        self.batch_sz = batch_sz
        self.latent_dim = latent_dim
        self.n_classes = n_classes

        self.generator = self.create_generator()
        self.discriminator = self.create_discriminator()
        self.gan = self.define_gan()

        if not os.path.exists('./training_checkpoints/mnist-acgan'):
            os.makedirs('./training_checkpoints/mnist-acgan')
        self.checkpoint_prefix = './training_checkpoints/mnist-acgan'



    def create_discriminator(self):
        # image input
        in_image = layers.Input(shape=[28,28,1])
        #first convolution
        fe = layers.Conv2D(64, (5,5), strides=(2,2), padding='same')(in_image)
        fe = layers.LeakyReLU()(fe)
        fe = layers.Dropout(0.3)(fe)
        #second convolution
        fe = layers.Conv2D(128, (5,5), padding='same')(fe)
        fe = layers.LeakyReLU()(fe)
        fe = layers.Dropout(0.3)(fe)
        #third convolution
        fe = layers.Conv2D(256, (5,5), strides=(2,2), padding='same')(fe)
        fe = layers.LeakyReLU()(fe)
        fe = layers.Dropout(0.3)(fe)
        #last layer
        fe = layers.Flatten()(fe)
        # real/fake output
        out1 = layers.Dense(1, activation='sigmoid')(fe)
        # class label output
        out2 = layers.Dense(self.n_classes, activation='softmax')(fe)
        # define model
        model = tf.keras.Model(in_image, [out1, out2])
        # compile model
        opt = tf.keras.optimizers.Adam(lr=0.0002, beta_1=0.5)
        model.compile(loss=['binary_crossentropy', 'sparse_categorical_crossentropy'], optimizer=opt)
        return model

    def create_generator(self):
        # label input
        in_label = layers.Input(shape=(1,))
        # embedding for categorical input
        li = layers.Embedding(self.n_classes, 50)(in_label)
        # linear multiplication
        n_nodes = 7 * 7
        li = layers.Dense(n_nodes)(li)
        # reshape to additional channel
        li = layers.Reshape((7, 7, 1))(li)
        # image generator input
        in_lat = layers.Input(shape=(self.latent_dim,))
        # foundation for 7x7 image
        n_nodes = 256 * 7 * 7
        gen = layers.Dense(n_nodes, use_bias=False)(in_lat)
        gen = layers.BatchNormalization()(gen)
        gen = layers.Activation('relu')(gen)
        gen = layers.Reshape((7, 7, 256))(gen)
        # merge image gen and label input
        merge = layers.Concatenate()([gen, li])
        # upsample
        gen = layers.Conv2DTranspose(128, (5,5), strides=(1,1), padding='same', use_bias=False)(merge)
        gen = layers.BatchNormalization()(gen)
        gen = layers.Activation('relu')(gen)
        # upsample
        gen = layers.Conv2DTranspose(64, (5,5), strides=(2,2), padding='same', use_bias=False)(gen)
        gen = layers.BatchNormalization()(gen)
        gen = layers.Activation('relu')(gen)

        gen = layers.Conv2DTranspose(1, (5,5), strides=(2,2), padding='same', use_bias=False, activation='tanh')(gen)

        # define model
        model = tf.keras.Model([in_lat, in_label], gen)
        return model

    # define the combined generator and discriminator model, for updating the generator
    def define_gan(self):
        # make weights in the discriminator not trainable
        self.discriminator.trainable = False
        # connect the outputs of the generator to the inputs of the discriminator
        gan_output = self.discriminator(self.generator.output)
        # define gan model as taking noise and label and outputting real/fake and label outputs
        model = tf.keras.Model(self.generator.input, gan_output)
        # compile model
        opt = tf.keras.optimizers.Adam(lr=0.0002, beta_1=0.5)
        model.compile(loss=['binary_crossentropy', 'sparse_categorical_crossentropy'], optimizer=opt)
        return model

    # generate points in latent space as input for the generator
    def generate_latent_points(self, n_samples):
        # generate points in the latent space
        z_input = np.random.randn(n_samples, self.latent_dim)
        # generate labels
        labels = np.random.randint(0, self.n_classes, n_samples)
        return [z_input, labels]

    # use the generator to generate n fake examples, with class labels
    def generate_fake_samples(self, n_samples):
        # generate points in latent space
        z_input, labels_input = self.generate_latent_points(n_samples)
        # predict outputs
        images = self.generator.predict([z_input, labels_input])
        # create class labels
        y = np.random.uniform(0, 0.3, (n_samples, 1))
        return [images, labels_input], y


    # generate points in latent space as input for the generator
    def generate_latent_points_class(self, n_samples, n_class):
        # generate points in the latent space
        z_input = np.random.randn(n_samples, self.latent_dim)
        # generate labels
        labels = np.asarray([n_class for _ in range(n_samples)])
        return [z_input, labels]
 

    # create and save a plot of generated images
    def save_plot(self,examples, n_examples, filename):
        # plot images
        for i in range(n_examples):
            # define subplot
            plt.subplot(sqrt(n_examples), sqrt(n_examples), 1 + i)
            # turn off axis
            plt.axis('off')
            # plot raw pixel data
            plt.imshow(examples[i, :, :, 0], cmap="gray")
        plt.savefig(filename)


    def train(self, dataset):
        d_real_losses = []
        d_fake_losses = []
        g_losses = []
        for epoch in range(self.epochs):
            start = time.time()

            #TODO this needs a better solution
            previous = None
            #k=0
            for image_batch in dataset:
                X_real, labels_real = image_batch
                #solve last batch of epoch size issue
                if X_real.shape[0] != self.batch_sz:
                    diff = self.batch_sz - X_real.shape[0] + 1
                    X_prior, labels_prior = previous
                    X_real = np.concatenate((X_real, X_prior[-diff:-1]))
                    labels_real = np.concatenate((labels_real, labels_prior[-diff:-1]))
                #train on batch here
                # update discriminator model weights
                y_real = np.random.uniform(0.7, 1, (self.batch_sz, 1))
                d_real_loss = self.discriminator.train_on_batch(X_real, [y_real, labels_real])
                
                # generate 'fake' examples
                [X_fake, labels_fake], y_fake = self.generate_fake_samples(self.batch_sz)
                # update discriminator model weights
                d_fake_loss = self.discriminator.train_on_batch(X_fake, [y_fake, labels_fake])
                
                # prepare points in latent space as input for the generator
                [z_input, z_labels] = self.generate_latent_points(self.batch_sz)
                # create inverted labels for the fake samples
                y_gan = np.random.uniform(0.7, 1, (self.batch_sz, 1))
                # update the generator via the discriminator's error
                g_loss = self.gan.train_on_batch([z_input, z_labels], [y_gan, z_labels])
                
                previous = image_batch
                #if k > 791:
                #    break
                #k += 1

            d_real_losses.append(d_real_loss)
            d_fake_losses.append(d_fake_loss)
            g_losses.append(g_loss)

            # Save the model every epoch
            if (epoch + 1) % 10 == 0:
                #self.checkpoint.save(file_prefix = self.checkpoint_prefix)
                self.generator.save('{}/generator-e{}.h5'.format(self.checkpoint_prefix, epoch+1))
                self.discriminator.save('{}/discriminator-e{}.h5'.format(self.checkpoint_prefix, epoch+1))
                for i in range(self.n_classes):
                    if not os.path.exists('./plots/mnist-acgan/epoch{}'.format(epoch+1)):
                        os.makedirs('./plots/mnist-acgan/epoch{}'.format(epoch+1))
                    filename = './plots/mnist-acgan/epoch{}/mnist-acgan_samples_class_{}.png'.format(epoch+1,i)
                    latent_points, labels = self.generate_latent_points_class(16, i)
                    X = self.generator.predict([latent_points, labels])
                    self.save_plot(X, 16, filename)
            print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))

        self.generator.save('{}/generator-e{}.h5'.format(self.checkpoint_prefix, self.epochs))
        self.discriminator.save('{}/discriminator-e{}.h5'.format(self.checkpoint_prefix, self.epochs))
        return d_fake_losses, d_real_losses, g_losses


if __name__ == "__main__":
    dataset, test_images, test_labels = util.get_mnist()
    acgan = ACGAN(epochs=50, batch_sz=256, n_classes=10)
    d_fake_losses, d_real_losses, g_losses = acgan.train(dataset)
    _eval = acgan.discriminator.evaluate(test_images, [np.ones(10000,), test_labels])
    with open("mnist-acgan-d_fake_losses.txt", "wb") as fp:
        pickle.dump(d_fake_losses, fp)
    with open("mnist-acgan-d_real_losses.txt", "wb") as fp:
        pickle.dump(d_real_losses, fp)
    with open("mnist-acgan-g_losses.txt", "wb") as fp:
        pickle.dump(g_losses, fp)
    with open("mnist-acgan-d_fake_losses.txt", "wb") as fp:
        pickle.dump(_eval, fp)