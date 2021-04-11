import glob
#tensorlfow
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import Progbar
# utils
import inspect
import pickle
import time
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
from math import sqrt
from scipy.stats import truncnorm
# modules
from OriginalOps.spectral_normalization import SpectralConv2D, SpectralDense
from OriginalOps.ops import ResnetBlock, ResnetBlockUp, ResnetBlockDown
from OriginalOps.attention import Attention
from OriginalOps.global_sum_pooling import GlobalSumPooling2D
from OriginalOps.conditional_batch_normalization import ConditionalBatchNormalization

from functools import partial

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
        plt.imshow(examples[i, :, :, :])
    plt.savefig(filename)


class ACGAN():
    def __init__(self, truncation=2.0, latent_dim=128, n_classes=10, model_name="acgan", discriminator=None, generator=None):
        self.truncation = truncation
        self.latent_dim = latent_dim
        self.n_classes = n_classes
        self.model_name = model_name
        #setup paths
        if not os.path.exists('./history/{}'.format(model_name)):
            os.makedirs('./history/{}'.format(model_name))
        self.history_prefix = './history/{}'.format(self.model_name)

        if not os.path.exists('{}/evaluation'.format(self.history_prefix)):
            os.makedirs('{}/evaluation'.format(self.history_prefix))
        self.evaluation_prefix = '{}/evaluation'.format(self.history_prefix)

        if not os.path.exists('{}/plots'.format(self.history_prefix)):
            os.makedirs('{}/plots'.format(self.history_prefix))
        self.plots_prefix = '{}/plots'.format(self.history_prefix)

        if not os.path.exists('{}/training_checkpoints'.format(self.history_prefix)):
            os.makedirs('{}/training_checkpoints'.format(self.history_prefix))
        self.training_checkpoints_prefix = '{}/training_checkpoints'.format(self.history_prefix)
        #create/load models
        if generator != None:
            self.generator = generator
        else:
            self.generator = self.create_generator()
        if discriminator != None:
            self.discriminator = discriminator
        else:
            self.discriminator = self.create_discriminator()
        self.gan = self.define_gan()
        #save discriminator and generator in files :)
        D_lines = inspect.getsource(self.create_discriminator)
        G_lines = inspect.getsource(self.create_generator)
        C_lines = inspect.getsource(self.define_gan)
        with open('{}/models.txt'.format(self.history_prefix), 'w') as f:
            f.write(D_lines)
            f.write('\n')
            f.write(G_lines)
            f.write('\n')
            f.write(C_lines)
            f.close()
        

    def create_discriminator(self):
        init = tf.keras.initializers.Orthogonal()
        # image input
        in_image = layers.Input(shape=[32,32,3])
        # ResBlocks
        D = ResnetBlockDown(in_image, 96)
        D = Attention(D, epsilon=1.0e-8)
        D = ResnetBlockDown(D, 192)
        D = ResnetBlock(D, 384)
        D = ResnetBlock(D, 384)
        D = layers.LeakyReLU(alpha=0.2)(D)
        D = GlobalSumPooling2D()(D)
        #last layer
        D = layers.Flatten()(D)
        # real/fake output
        out1 = SpectralDense(1, epsilon=1.0e-8, kernel_initializer=init, activation='sigmoid', name='out_fake')(D)
        # class label output
        out2 = SpectralDense(self.n_classes, epsilon=1.0e-8, kernel_initializer=init, activation='softmax', name='out_aux')(D)
        # define model
        model = tf.keras.Model(in_image, [out1, out2], name="discriminator")
        # compile model
        opt = tf.keras.optimizers.Adam(lr=0.0002, beta_1=0.0, beta_2=0.999 ,epsilon=1.0e-8)
        model.compile(
            loss=['hinge', 'sparse_categorical_crossentropy'], 
            optimizer=opt,
            metrics={'out_fake': 'accuracy', 'out_aux': tf.keras.metrics.SparseCategoricalAccuracy()})
        model.summary()
        return model

    def create_generator(self):
        init = tf.keras.initializers.Orthogonal()
        # label input
        z = layers.Input(shape=(self.latent_dim,))
        # this will be our label
        y = layers.Input(shape=(1,), dtype='float32')

        y_emb = SpectralDense(self.latent_dim, use_bias=False, epsilon=1.0e-8, kernel_initializer=init)(y)
        c = layers.Concatenate()([z, y_emb])
        G = SpectralDense(4*4*384, use_bias=False, epsilon=1.0e-8, kernel_initializer=init)(c)
        G = layers.Reshape((4,4,384))(G)
        # ResBlocks
        G = ResnetBlockUp(G, c, 384)
        G = ResnetBlockUp(G, c, 192)
        G = Attention(G, epsilon=1.0e-8)
        G = ResnetBlockUp(G, c, 96)
        # End part
        G = ConditionalBatchNormalization(G, c, momentum=0.99, epsilon=1.0e-8)
        G = layers.LeakyReLU(alpha=0.2)(G)
        G = SpectralConv2D(3, 3, epsilon=1.0e-8, padding="same", kernel_initializer=init)(G)
        G = layers.Activation("tanh")(G)
        # define model
        model = tf.keras.Model([z, y], G, name="generator")
        model.summary()
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
        opt = tf.keras.optimizers.Adam(lr=0.00005, beta_1=0.0, beta_2=0.999, epsilon=1.0e-8)
        model.compile(loss=['hinge', 'sparse_categorical_crossentropy'], optimizer=opt)
        return model


    def generate_real_samples(self, train_x, train_y, n_samples, index):
        # select images and labels
        X = train_x[index*n_samples:(index+1)*n_samples]
        labels = train_y[index*n_samples:(index+1)*n_samples]
        # generate class labels
        y = np.ones((n_samples, 1))
        return [X, labels], y

    # generate points in latent space as input for the generator
    def generate_latent_points(self, n_samples):
        # generate points in the latent space
        z_input = truncnorm.rvs(-2, 2, size=(n_samples, self.latent_dim), random_state=None).astype(np.float32)
        z_input = z_input * self.truncation
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
        y = np.zeros((n_samples, 1))
        return [images, labels_input], y


    # generate points in latent space as input for the generator
    # n_sanples  should be a power of self.n_classes
    def generate_latent_points_all_classes(self, n_samples):
        # generate points in the latent space
        z_input = truncnorm.rvs(-2, 2, size=(n_samples, self.latent_dim), random_state=None).astype(np.float32)
        z_input = z_input * self.truncation
        # generate labels
        n = int(sqrt(n_samples))
        labels = np.asarray([c for c in range(n) for _ in range(n)])
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
            plt.imshow(examples[i, :, :, :])
        plt.savefig(filename)


    def train(self, train_x, train_y, batches_per_epoch, epochs=100, batch_size=32):
        d_real_losses = []
        d_fake_losses = []
        g_losses = []

        if not batches_per_epoch: 
            batches_per_epoch = train_x.shape[0]//batch_size

        for epoch in range(epochs):
            print('Starting epoch {}'.format(epoch+1))
            start = time.time()
            progress_bar = Progbar(target=batches_per_epoch)
            for i in range(batches_per_epoch):
                progress_bar.update(i)
                #train D more times than G
                for _ in range(2):
                    #train on batch here
                    [X_real, labels_real], y_real = self.generate_real_samples(train_x, train_y, batch_size, i)
                    # update discriminator model weights
                    d_real_loss = self.discriminator.train_on_batch(X_real, [y_real, labels_real], return_dict=True)
                    
                    # generate 'fake' examples
                    [X_fake, labels_fake], y_fake = self.generate_fake_samples(batch_size)
                    # update discriminator model weights
                    d_fake_loss = self.discriminator.train_on_batch(X_fake, [y_fake, labels_real], return_dict=True)
                
                # prepare points in latent space as input for the generator
                [z_input, z_labels] = self.generate_latent_points(batch_size)
                # create inverted labels for the fake samples
                y_gan = np.ones((batch_size, 1))
                # update the generator via the discriminator's error
                g_loss = self.gan.train_on_batch([z_input, z_labels], [y_gan, z_labels], return_dict=True)
            
            print('\nD_Real_loss: {};\nD_Fake_loss: {};\nG_loss:      {}; \nTime for epoch {} is {} sec'.format(d_real_loss,d_fake_loss,g_loss,epoch+1,time.time()-start))

            d_real_losses.append(d_real_loss)
            d_fake_losses.append(d_fake_loss)
            g_losses.append(g_loss)

            # Save the model every x epochs
            if (epoch + 1) % 10 == 0 or  (epoch + 1) < 10:
                self.generator.save('{}/generator-e{}.h5'.format(self.training_checkpoints_prefix, epoch+1))
                self.discriminator.save('{}/discriminator-e{}.h5'.format(self.training_checkpoints_prefix, epoch+1))

                # generate an image of 100 fake images (10 per class)
                filename = '{}/e{}-acgan_samples.png'.format(self.plots_prefix, epoch+1)
                latent_points, labels = self.generate_latent_points_all_classes(100)
                X = self.generator.predict([latent_points, labels])
                X = (X+1) / 2
                self.save_plot(X, 100, filename)

                with open("{}/d_fake_losses.p".format(self.evaluation_prefix), "wb") as fp:
                    pickle.dump(d_fake_losses, fp)
                with open("{}/d_real_losses.p".format(self.evaluation_prefix), "wb") as fp:
                    pickle.dump(d_real_losses, fp)
                with open("{}/g_losses.p".format(self.evaluation_prefix), "wb") as fp:
                    pickle.dump(g_losses, fp)

        self.generator.save('{}/generator-e{}.h5'.format(self.training_checkpoints_prefix, epochs))
        self.discriminator.save('{}/discriminator-e{}.h5'.format(self.training_checkpoints_prefix, epochs))
        return d_fake_losses, d_real_losses, g_losses


if __name__ == "__main__":
    batch_size = 512
    (train_x, train_y), (test_x, test_y) = cifar10.load_data()
    
    #val_size = 5000
    # take val_size images from train to use as validation
    #val_x = train_x[-val_size:]
    #val_y = train_y[-val_size:]
    # and now remove them from the train data
    #train_x = train_x[:-val_size]
    #train_y = train_y[:-val_size]

    train_x = train_x.astype('float32')
    train_x = (train_x - 127.5) / 127.5
    # automatically create new model name from existing folders
    base_name = 'bigacganO-cifar10-'
    numbers = [int(name.split('-')[-1]) for name in os.listdir("./history") if os.path.isdir('./history/'+name) and len(name.split('-'))==3]
    name = base_name + str(numbers[-1]+1)
    # create model
    acgan = ACGAN(truncation=2.0, n_classes=10, model_name=name)
    d_fake_losses, d_real_losses, g_losses = acgan.train(train_x, train_y, epochs=500, batch_size=batch_size, batches_per_epoch=train_x.shape[0]//batch_size)
