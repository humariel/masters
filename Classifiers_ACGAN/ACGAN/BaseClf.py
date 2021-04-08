import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.datasets import cifar10
import pickle

def create_discriminator():
    init = tf.keras.initializers.RandomNormal()
    # image input
    in_image = layers.Input(shape=[32,32,3])
    #Gaussian Noise to help with overfit
    D = layers.GaussianNoise(0.15)(in_image)
    #convolution
    D = layers.Conv2D(64, (5,5), strides=(2,2), padding='same', kernel_initializer=init)(D)
    D = layers.LeakyReLU(alpha=0.2)(D)
    D = layers.Dropout(0.3)(D)
    #convolution
    D = layers.Conv2D(128, (5,5), strides=(2,2), padding='same', kernel_initializer=init)(D)
    D = layers.LeakyReLU(alpha=0.2)(D)
    D = layers.Dropout(0.3)(D)
    #convolution
    D = layers.Conv2D(256, (5,5), strides=(2,2), padding='same', kernel_initializer=init)(D)
    D = layers.LeakyReLU(alpha=0.2)(D)
    D = layers.Dropout(0.3)(D)
    #convolution
    D = layers.Conv2D(512, (5,5), strides=(2,2), padding='same', kernel_initializer=init)(D)
    D = layers.LeakyReLU(alpha=0.2)(D)
    D = layers.Dropout(0.3)(D)
    #last layer
    D = layers.Flatten()(D)
    # class label output
    out = layers.Dense(10, activation='softmax', name='out_aux')(D)
    # define model
    model = tf.keras.Model(in_image, out, name="discriminator")
    # compile model
    opt = tf.keras.optimizers.Adam(lr=0.0002, beta_1=0.5)
    model.compile(
        loss=['sparse_categorical_crossentropy'], 
        optimizer=opt,
        metrics={'out_aux': tf.keras.metrics.SparseCategoricalAccuracy()})
    model.summary()
    return model

model = create_discriminator()

# load cifar-10
(train_x, train_y), (test_x, test_y) = cifar10.load_data()

history = model.fit(train_x, train_y,
                    batch_size=64,
                    epochs=200)

print(history.history)
with open('baseClf.p', "wb") as f_history:
    pickle.dump(history.history, f_history)

model.save('baseClf.h5')
