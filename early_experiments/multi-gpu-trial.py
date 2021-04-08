# Copyright 2019 Zuru Tech HK Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Adversarial trainer example."""

import tensorflow as tf
from tensorflow import keras  # pylint: disable=no-name-in-module
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.models import load_model

from ashpy.losses import DiscriminatorMinMax, GeneratorBCE
from ashpy.metrics import InceptionScore
from ashpy.models.gans import ConvDiscriminator, ConvGenerator
from ashpy.trainers import AdversarialTrainer

import util

# define the standalone discriminator model
def define_discriminator(in_shape=(256,256,3), n_classes=2):
	# label input
	in_label = Input(shape=(1,))
	# embedding for categorical input
	li = Embedding(n_classes, 50)(in_label)
	# scale up to image dimensions with linear activation
	n_nodes = in_shape[0] * in_shape[1]
	li = Dense(n_nodes)(li)
	# reshape to additional channel
	li = Reshape((in_shape[0], in_shape[1], 1))(li)
	# image input
	in_image = Input(shape=in_shape)
	# concat label as a channel
	merge = Concatenate()([in_image, li])
	# downsample
	fe = Conv2D(128, (3,3), strides=(2,2), padding='same')(merge)
	fe = LeakyReLU(alpha=0.2)(fe)
	# downsample
	fe = Conv2D(128, (3,3), strides=(2,2), padding='same')(fe)
	fe = LeakyReLU(alpha=0.2)(fe)
	# flatten feature maps
	fe = Flatten()(fe)
	# dropout
	fe = Dropout(0.4)(fe)
	# output
	out_layer = Dense(1, activation='sigmoid')(fe)
	# define model
	model = Model([in_image, in_label], out_layer)
	return model

# define the standalone generator model
def define_generator(latent_dim, n_classes=2):
	# label input
	in_label = Input(shape=(1,))
	# embedding for categorical input
	li = Embedding(n_classes, 50)(in_label)
	# linear multiplication
	n_nodes = 64 * 64
	li = Dense(n_nodes)(li)
	# reshape to additional channel
	li = Reshape((64, 64, 1))(li)
	# image generator input
	in_lat = Input(shape=(latent_dim,))
	# foundation for 7x7 image
	n_nodes = 128 * 64 * 64
	gen = Dense(n_nodes)(in_lat)
	gen = LeakyReLU(alpha=0.2)(gen)
	gen = Reshape((64, 64, 128))(gen)
	# merge image gen and label input
	merge = Concatenate()([gen, li])
	# upsample to 128*128
	gen = Conv2DTranspose(128, (4,4), strides=(2,2), padding='same')(merge)
	gen = LeakyReLU(alpha=0.2)(gen)
	# upsample to 256*256
	gen = Conv2DTranspose(128, (4,4), strides=(2,2), padding='same')(gen)
	gen = LeakyReLU(alpha=0.2)(gen)
	# output
	out_layer = Conv2D(3, (7,7), activation='tanh', padding='same')(gen)
	# define model
	model = Model([in_lat, in_label], out_layer)
	return model


def main():
    """Adversarial trainer example."""
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        latent_dim = 100
        generator = define_generator(latent_dim)

        discriminator = define_discriminator()

        # Losses
        generator_bce = GeneratorBCE()
        minmax = DiscriminatorMinMax()

        # Trainer
        logdir = "log/adversarial"

        # InceptionScore: keep commented until the issues
        # https://github.com/tensorflow/tensorflow/issues/28599
        # https://github.com/tensorflow/hub/issues/295
        # Haven't been solved and merged into tf2

        metrics = [
            # InceptionScore(
            #    InceptionScore.get_or_train_inception(
            #        mnist_dataset,
            #        "mnist",
            #        num_classes=10,
            #        epochs=1,
            #        fine_tuning=False,
            #        logdir=logdir,
            #    ),
            #    model_selection_operator=operator.gt,
            #    logdir=logdir,
            # )
        ]

        epochs = 50
        trainer = AdversarialTrainer(
            generator=generator,
            discriminator=discriminator,
            generator_optimizer=tf.optimizers.Adam(2e-4),
            discriminator_optimizer=tf.optimizers.Adam(2e-4),
            generator_loss=generator_bce,
            discriminator_loss=minmax,
            epochs=epochs,
            metrics=metrics,
            logdir=logdir,
        )

        batch_size = 32

        images = './isic2019/ISIC_2019_Training_Input/'
        gt = './isic2019/ISIC_2019_Training_GroundTruth_Original.csv'

        data = util.load_isic_2_class(images,gt,batch_size)
        # Add noise in the same dataset, just by mapping.
        # The return type of the dataset must be: tuple(tuple(a,b), noise)
        dataset = data.map(
            lambda x, y: ((x, y), tf.random.normal(shape=(batch_size, latent_dim)))
        )

        trainer(dataset,1,0)


if __name__ == "__main__":
    main()