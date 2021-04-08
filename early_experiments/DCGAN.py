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
from ashpy.trainers import AdversarialTrainer
from ashpy.losses import DiscriminatorMinMax, GeneratorBCE

class DCGAN():
    def __init__(self, epochs=50, batch_sz=256, noise_dim=100, samples_to_gen=16):
        self.epochs = epochs
        self.batch_sz = batch_sz
        self.noise_dim = noise_dim
        self.samples_to_gen = samples_to_gen

        self.generator = self.create_generator()
        self.discriminator = self.create_discriminator()

        self.generator_optimizer = tf.keras.optimizers.Adam(1e-4)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

        if not os.path.exists('./training_checkpoints/dcgan'):
            os.makedirs('./training_checkpoints/dcgan')

        self.checkpoint_prefix = './training_checkpoints/dcgan'
        """checkpoint_dir = './training_checkpoints/'
        self.checkpoint_prefix = os.path.join(checkpoint_dir, "dcgan")
        self.checkpoint = tf.train.Checkpoint(generator_optimizer=self.generator_optimizer,
                                 discriminator_optimizer=self.discriminator_optimizer,
                                 generator=self.generator,
                                 discriminator=self.discriminator)"""


    def set_generator(self, generator):
        self.generator = generator


    def set_discriminator(self, discriminator):
        self.discriminator = discriminator


    def create_generator(self):
        model = tf.keras.Sequential()
        #initial Dense(FC) layer. We start with a 100 unit vector as input and 
        #upsmaple to a 7*7*256
        model.add(layers.Dense(7*7*256, use_bias=False, input_shape=(100,)))
        model.add(layers.BatchNormalization())
        model.add(layers.ReLU())

        model.add(layers.Reshape((7,7,256)))
        assert model.output_shape == (None, 7, 7, 256) # Note: None is the batch size

        #add one transposed convolution (upsample)
        model.add(layers.Conv2DTranspose(128, (5,5), strides=(1,1), padding='same', use_bias=False))
        #confirm output shape
        assert model.output_shape == (None, 7, 7, 128)
        model.add(layers.BatchNormalization())
        model.add(layers.ReLU())

        #one more transposed convolution (upsample)
        model.add(layers.Conv2DTranspose(64, (5,5), strides=(2,2), padding='same', use_bias=False))
        #confirm output shape
        assert model.output_shape == (None, 14, 14, 64)
        model.add(layers.BatchNormalization())
        model.add(layers.ReLU())

        #final layer, using a tanh activation
        model.add(layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
        assert model.output_shape == (None, 28, 28, 1)

        return model


    def generator_loss(self, fake_output):
        cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        return cross_entropy(tf.ones_like(fake_output), fake_output)


    def create_discriminator(self):
        model = tf.keras.Sequential()
        #first convolution
        model.add(layers.Conv2D(64, (5,5), strides=(2,2), padding='same', input_shape=[28,28,1]))
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(0.3))
        #second convolution
        model.add(layers.Conv2D(128, (5,5), strides=(2,2), padding='same'))
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(0.3))
        #third convolution
        model.add(layers.Conv2D(256, (5,5), strides=(1,1), padding='same'))
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(0.3))
        #last layer
        model.add(layers.Flatten())
        model.add(layers.Dense(1))

        return model


    def discriminator_loss(self, real_output, fake_output):
        #normal binary cross entropy
        cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

        real_loss = cross_entropy(tf.ones_like(real_output), real_output)
        fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
        loss = real_loss + fake_loss

        return loss


    @tf.function
    def train_step(self, images):
        noise = tf.random.normal([self.batch_sz, self.noise_dim])

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_imgs = self.generator(noise, training=True)

            real_output = self.discriminator(images, training=True)
            fake_output = self.discriminator(generated_imgs, training=True)

            gen_loss = self.generator_loss(fake_output)
            disc_loss = self.discriminator_loss(real_output, fake_output)

        gen_deltas = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        disc_deltas = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)

        self.generator_optimizer.apply_gradients(zip(gen_deltas, self.generator.trainable_variables))
        self.discriminator_optimizer.apply_gradients(zip(disc_deltas, self.discriminator.trainable_variables))


    def train(self, dataset):
        for epoch in range(self.epochs):
            start = time.time()

            for image_batch in dataset:
                self.train_step(image_batch)

            # Save the model every 15 epochs
            if (epoch + 1) % 25 == 0:
                #self.checkpoint.save(file_prefix = self.checkpoint_prefix)
                self.generator.save('{}/generator-e{}.h5'.format(self.checkpoint_prefix, epoch))
                self.discriminator.save('{}/discriminator-e{}.h5'.format(self.checkpoint_prefix, epoch))
                filename = './plots/dcgan/dcgan_gen_sample_minst_{:04d}.png'.format(epoch)
                self.generate_and_save_images(self.generator, filename, tf.random.normal([self.samples_to_gen, self.noise_dim]))
            print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))

        self.generator.save('{}/generator-e{}.h5'.format(self.checkpoint_prefix, self.epochs))
        self.discriminator.save('{}/discriminator-e{}.h5'.format(self.checkpoint_prefix, self.epochs))
            


    def generate_and_save_images(self, model, filename, test_input):
        predictions = model(test_input, training=False)

        fig = plt.figure(figsize=(4,4))

        for i in range(predictions.shape[0]):
            plt.subplot(4, 4, i+1)
            plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
            plt.axis('off')

        plt.savefig(filename)


if __name__ == '__main__':
    dataset,_,_ = util.get_mnist()
    gan = DCGAN(epochs=50)
    gan.train(dataset)

    # gan = DCGAN()
    # gan.set_generator(tf.keras.models.load_model('./training_checkpoints/dcgan/generator-e50.h5', compile=False))
    # gan.set_discriminator(tf.keras.models.load_model('./training_checkpoints/dcgan/discriminator-e50.h5', compile=False))

    # gan.generate_and_save_images(gan.generator, 1, tf.random.normal([16, 100]))

