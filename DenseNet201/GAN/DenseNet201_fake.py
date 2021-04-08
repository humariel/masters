## BASE MODEL AND DATA AUGMENTED WITH STYLEGAN2-ADA ##

import tensorflow as tf 
from tensorflow.keras.datasets import cifar10
import pickle
import os 
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import random
from sklearn.model_selection import train_test_split
import argparse

# GENERATED IMAGES
BASE_DIR = os.getcwd() + "/../Datasets"
CIFAR_FAKE_DIR = BASE_DIR + "/cifar-10-generated/"

BATCH_SIZE = 32
IMG_SIZE = (224, 224)

def main(halve):

    if halve:
        model_name = 'densenet_fake_halved.h5'
        history_name = "history_fake_halved.p"
    else:
        model_name = 'densenet_fake.h5'
        history_name = "history_fake.p"

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
    print(train_x.shape)
    print(train_y.shape)
    # shuffle x and y in the same way
    shuffle = np.random.permutation(len(train_x))
    train_x = train_x[shuffle]
    train_y = train_y[shuffle]

    """
    print(len(train_x), len(train_y))
    print(len(val_x), len(val_y))
    print(len(test_x), len(test_y))
    """

    IMG_SHAPE = IMG_SIZE + (3,)

    densenet = tf.keras.applications.DenseNet201(
        include_top=False, weights='imagenet', input_tensor=None, input_shape=IMG_SHAPE, pooling=None)

    densenet.trainable = True
    initializer = tf.keras.initializers.he_normal(seed=32)

    inputs = tf.keras.Input(shape=(32,32,3))
    x = tf.keras.applications.densenet.preprocess_input(inputs)
    x = tf.keras.layers.Lambda(lambda image: tf.image.resize(image, (224, 224)))(x)
    x = densenet(x, training=True)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(512, activation='relu')(x)
    outputs = tf.keras.layers.Dense(10, activation='softmax')(x)
    model = tf.keras.Model(inputs, outputs)

    model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.0001),
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
    model.summary()

    early_stop = tf.keras.callbacks.EarlyStopping(monitor='loss', min_delta=0.001, patience=10)

    history = model.fit(train_x, train_y,
                        batch_size=BATCH_SIZE,
                        epochs=100,
                        validation_data=(val_x,val_y),
                        validation_steps=10,
                        callbacks=[early_stop])
    with open(history_name, "wb") as f_history:
        pickle.dump(history.history, f_history)

    model.save(model_name)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DenseNet trainer')
    parser.add_argument('--halve', default=False, action='store_true')
    args = parser.parse_args()

    main(args.halve)
