import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.datasets import cifar10
import pickle

def create_discriminator():
    # image input
    in_image = layers.Input(shape=[32,32,3])
    #convolution
    D = layers.Conv2D(64, (5,5), strides=(2,2), padding='same', kernel_initializer='glorot_normal')(in_image)
    D = layers.LeakyReLU(alpha=0.2)(D)
    D = layers.Dropout(0.5)(D)
    #convolution
    D = layers.Conv2D(128, (5,5), strides=(2,2), padding='same', kernel_initializer='glorot_normal')(D)
    D = layers.LeakyReLU(alpha=0.2)(D)
    D = layers.Dropout(0.5)(D)
    #convolution
    D = layers.Conv2D(256, (5,5), strides=(2,2), padding='same', kernel_initializer='glorot_normal')(D)
    D = layers.LeakyReLU(alpha=0.2)(D)
    D = layers.Dropout(0.5)(D)
    #convolution
    D = layers.Conv2D(512, (5,5), strides=(2,2), padding='same', kernel_initializer='glorot_normal')(D)
    D = layers.LeakyReLU(alpha=0.2)(D)
    D = layers.Dropout(0.5)(D)
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
with open('BaseClf.p', "wb") as f_history:
    pickle.dump(history.history, f_history)

model.save('BaseClf.h5')
