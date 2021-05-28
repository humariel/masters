import tensorflow as tf
from tensorflow.keras import layers
import tensorflow_addons as tfa

from Ops.ops import ResnetBlock, ResnetBlockUp, ResnetBlockDown
from Ops.attention import Attention
from Ops.global_sum_pooling import GlobalSumPooling2D
from Ops.conditional_batch_normalization import ConditionalBatchNormalization


from util import *

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)



def create_discriminator():
    init = tf.keras.initializers.GlorotNormal()
    # image input
    in_image = layers.Input(shape=[128,128,3])

    D = layers.Conv2D(128, 5, padding="same", kernel_initializer=init)(in_image)
    # ResBlocks
    D = ResnetBlockDown(D, 128)
    D = Attention(D, epsilon=1.0e-8)
    D = ResnetBlockDown(D, 128)
    D = ResnetBlockDown(D, 128)
    D = ResnetBlockDown(D, 128)
    D = ResnetBlockDown(D, 128)
    D = ResnetBlock(D, 128)
    D = layers.LeakyReLU(alpha=0.2)(D)
    D = GlobalSumPooling2D()(D)
    #last layer
    D = layers.Flatten()(D)
    # class label output
    out = layers.Dense(1, kernel_initializer=init, activation='sigmoid', name='out_aux')(D)
    # define model
    model = tf.keras.Model(in_image, out, name="discriminator")
    # compile model
    opt = tf.keras.optimizers.Adam(lr=0.0002, beta_1=0.5, epsilon=1.0e-8)
    model.compile(
        loss=['binary_crossentropy'], 
        optimizer=opt,
        metrics={'out_aux': 'accuracy'})
    model.summary()
    return model

base_data_dir = "../../Datasets/SkinLesion_Mel_NonMel/"
train_data_dir = base_data_dir + "train/"
val_data_dir = base_data_dir + "val/"
test_data_dir = base_data_dir + "test/"

batch_size = 16
image_size = (128, 128)

train_gen, val_gen, _ = load_skin_lesion_dataset(train_data_dir, val_data_dir, test_data_dir, target_size=image_size, batch_size=batch_size)

model = create_discriminator()

history = model.fit(train_gen,
                    batch_size=batch_size,
                    epochs=250,
                    validation_data=val_gen
                    )

print(history.history)
with open('BaseClf.p', "wb") as f_history:
    pickle.dump(history.history, f_history)

model.save('BaseClf.h5')
