import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.datasets import cifar10

import pickle
import os
import numpy as np
from PIL import Image
import argparse

from Ops.spectral_normalization import SpectralConv2D, SpectralDense
from Ops.ops import ResnetBlock, ResnetBlockUp, ResnetBlockDown
from Ops.attention import Attention
from Ops.global_sum_pooling import GlobalSumPooling2D
from Ops.conditional_batch_normalization import ConditionalBatchNormalization

# GENERATED IMAGES
BASE_DIR = os.getcwd() + "/../../Datasets"
CIFAR_FAKE_DIR = BASE_DIR + "/cifar-10-generated_bigacgan/"

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
    opt = tf.keras.optimizers.Adam(lr=0.0002, beta_1=0.5, epsilon=1.0e-8)
    model.compile(
        loss=['sparse_categorical_crossentropy'], 
        optimizer=opt,
        metrics={'out_aux': tf.keras.metrics.SparseCategoricalAccuracy()})
    model.summary()
    return model


def main(halve):

    if halve:
        model_name = 'BigGANClf_fake2_halved.h5'
        history_name = "BigGANClf_fake2_halved.p"
    else:
        model_name = 'BigGANClf_fake2.h5'
        history_name = "BigGANClf_fake2.p"

    fakes = []
    fake_labels = []

    imgs_per_class = 2500 if halve else 5000
    for subdir, dirs, files in os.walk(CIFAR_FAKE_DIR):
        if dirs:
            continue
        else:
            label = int(subdir.split("/")[-1])
            size = len(files) if len(files) <= imgs_per_class else imgs_per_class
            labels = [label for x in range(0,size)]
            fake_labels += labels

            images = []
            count = 0
            for img in files:
                image = Image.open(subdir+"/"+img)
                data = np.asarray(image)
                images.append(data)
                count += 1
                if count >= size:
                    break

            fakes += images

    fake_train_x = np.asarray(fakes)
    fake_train_y = np.asarray(fake_labels, dtype=np.int8)
    fake_train_y = np.reshape(fake_train_y, (fake_train_y.shape[0],1))
    print(fake_train_x.shape)
    print(fake_train_y.shape)

    # load cifar-10
    (train_x, train_y), (test_x, test_y) = cifar10.load_data()

    val_size = 5000
    if halve:
        train_x, _, train_y, _ = train_test_split(train_x, train_y, test_size=0.5)
        test_x, _, test_y, _ = train_test_split(test_x, test_y, test_size=0.5)
        
        val_size = val_size // 2
        
    # take val_size images from train to use as validation
    val_x = train_x[-val_size:]
    val_y = train_y[-val_size:]
    # and now remove them from the train data
    train_x = train_x[:-val_size]
    train_y = train_y[:-val_size]

    #join generated to train data
    train_x = np.concatenate((train_x, fake_train_x))
    train_y = np.concatenate((train_y, fake_train_y))
    # shuffle x and y in the same way
    shuffle = np.random.permutation(len(train_x))
    train_x = train_x[shuffle]
    train_y = train_y[shuffle]
    train_x = train_x.astype('float32')
    train_x = (train_x - 127.5) / 127.5
    val_x = val_x.astype('float32')
    val_x = (val_x - 127.5) / 127.5

    model = create_discriminator()


    history = model.fit(train_x, train_y,
                        batch_size=64,
                        epochs=250,
                        validation_data=(val_x, val_y))

    print(history.history)
    with open(history_name, "wb") as f_history:
        pickle.dump(history.history, f_history)

    model.save(model_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DenseNet trainer')
    parser.add_argument('--halve', default=False, action='store_true')
    args = parser.parse_args()

    main(args.halve)
