import tensorflow as tf
from tensorflow.keras import layers
import numpy as np

#from .spectral_normalization import SpectralConv2D
#from .spectral_normalization import SpectralDense
from .attention import Attention
from .conditional_batch_normalization import ConditionalBatchNormalization

import  tensorflow_addons as tfa

def ResnetBlock(x, filters):
    init = tf.keras.initializers.GlorotNormal()
    y = layers.LeakyReLU(alpha=0.2)(x)
    y = layers.Dropout(0.5)(y)
    y = tfa.layers.SpectralNormalization(layers.Conv2D(filters, 5, padding="same", kernel_initializer=init))(y)
    y = layers.LeakyReLU(alpha=0.2)(y)
    y = layers.Dropout(0.5)(y)
    y = tfa.layers.SpectralNormalization(layers.Conv2D(filters, 5, padding="same", kernel_initializer=init))(y)

    z = tfa.layers.SpectralNormalization(layers.Conv2D(filters, 1, padding="same", kernel_initializer=init))(x)

    return layers.Add()([z,y])

def ResnetBlockDown(x, filters):
    init = tf.keras.initializers.GlorotNormal()
    y = layers.LeakyReLU(alpha=0.2)(x)
    y = layers.Dropout(0.5)(y)
    y = tfa.layers.SpectralNormalization(layers.Conv2D(filters, 5, padding="same", kernel_initializer=init))(y)
    y = layers.LeakyReLU(alpha=0.2)(y)
    y = layers.Dropout(0.5)(y)
    y = tfa.layers.SpectralNormalization(layers.Conv2D(filters, 5, padding="same", kernel_initializer=init))(y)
    y = layers.AveragePooling2D()(y)

    z = tfa.layers.SpectralNormalization(layers.Conv2D(filters, 1, padding="same", kernel_initializer=init))(x)
    z = layers.AveragePooling2D()(z)

    return layers.Add()([z,y])

def ResnetBlockUp(x, z, filters):
    init = tf.keras.initializers.RandomNormal()
    _x = x
    x = ConditionalBatchNormalization(x, z, momentum=0.99, epsilon=1.0e-8)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.UpSampling2D()(x)
    x = tfa.layers.SpectralNormalization(layers.Conv2D(filters, 3, padding="same", kernel_initializer=init))(x)
    x = ConditionalBatchNormalization(x, z, momentum=0.99, epsilon=1.0e-8)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = tfa.layers.SpectralNormalization(layers.Conv2D(filters, 3, padding="same", kernel_initializer=init))(x)

    _x = layers.UpSampling2D()(_x)
    _x = tfa.layers.SpectralNormalization(layers.Conv2D(filters, 1, padding="same", kernel_initializer=init))(_x)

    return layers.Add()([x,_x])