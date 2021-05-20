import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.datasets import cifar10

import pickle

from Ops.spectral_normalization import SpectralConv2D, SpectralDense
from Ops.ops import ResnetBlock, ResnetBlockUp, ResnetBlockDown
from Ops.attention import Attention
from Ops.global_sum_pooling import GlobalSumPooling2D
from Ops.conditional_batch_normalization import ConditionalBatchNormalization


def create_discriminator():
    init = tf.keras.initializers.GlorotNormal()
    # image input
    in_image = layers.Input(shape=[32,32,3])
    D = layers.Conv2D(128, 5, padding="same", kernel_initializer=init)(in_image)
    # ResBlocks
    D = ResnetBlockDown(D, 128)
    D = Attention(D, epsilon=1.0e-8)
    D = ResnetBlockDown(D, 128)
    D = ResnetBlock(D, 128)
    D = ResnetBlock(D, 128)
    D = layers.LeakyReLU(alpha=0.2)(D)
    D = GlobalSumPooling2D()(D)
    #last layer
    D = layers.Flatten()(D)
    # class label output
    out = SpectralDense(10, epsilon=1.0e-8, kernel_initializer=init, activation='softmax', name='out_aux')(D)
    # define model
    model = tf.keras.Model(in_image, out, name="discriminator")
    # compile model
    opt = tf.keras.optimizers.Adam(lr=0.0001, beta_1=0.5, epsilon=1.0e-8)
    model.compile(
        loss=['sparse_categorical_crossentropy'], 
        optimizer=opt,
        metrics={'out_aux': tf.keras.metrics.SparseCategoricalAccuracy()})
    model.summary()
    return model

model = create_discriminator()

# load cifar-10
(train_x, train_y), (test_x, test_y) = cifar10.load_data()

val_size = 5000
# take val_size images from train to use as validation
val_x = train_x[-val_size:]
val_y = train_y[-val_size:]
# and now remove them from the train data
train_x = train_x[:-val_size]
train_y = train_y[:-val_size]


train_x = train_x.astype('float32')
train_x = (train_x - 127.5) / 127.5
val_x = val_x.astype('float32')
val_x = (val_x - 127.5) / 127.5

history = model.fit(train_x, train_y,
                    batch_size=64,
                    epochs=250,
                    validation_data=(val_x, val_y))

print(history.history)
with open('BigGANClf.p', "wb") as f_history:
    pickle.dump(history.history, f_history)

model.save('BigGANClf.h5')
