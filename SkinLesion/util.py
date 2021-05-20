import tensorflow as tf
import numpy as np
from numpy import zeros
from numpy import ones
from numpy import concatenate
from numpy.random import randn
from numpy.random import randint
import pandas as pd
import os
import sklearn
import sklearn.metrics
from PIL import Image
import subprocess as sp

###############################DATASET UTILS#######################################
def load_skin_lesion_dataset(train_data_dir, val_data_dir, test_data_dir, target_size=(256,256), batch_size=64):

    def _preprocess(x):
        return (x-127.5)/127.5 # Normalize the images to [-1, 1]

    data_gen = tf.keras.preprocessing.image.ImageDataGenerator(preprocessing_function=_preprocess)

    train_gen = data_gen.flow_from_directory(
        train_data_dir, 
        target_size=target_size, 
        batch_size=batch_size, 
        class_mode="binary",
        )

    val_gen = data_gen.flow_from_directory(
        val_data_dir, 
        target_size=target_size, 
        batch_size=500, 
        class_mode="binary",
        )

    test_gen = data_gen.flow_from_directory(
        test_data_dir, 
        target_size=target_size, 
        batch_size=522, 
        class_mode="binary",
        )

    return train_gen, val_gen, test_gen

if __name__=='__main__':
    base_data_dir = "../Datasets/SkinLesion_Mel_NV/"
    train_data_dir = base_data_dir + "train/"
    val_data_dir = base_data_dir + "val/"
    test_data_dir = base_data_dir + "test/"

    batch_size = 64

    train_gen, val_gen, test_gen = load_skin_lesion_dataset(train_data_dir, val_data_dir, test_data_dir, batch_size=batch_size)

    """batches_per_epoch = train_gen.n//train_gen.batch_size

    for batch in range(batches_per_epoch):
        x,y = train_gen.next()
        y = y.reshape((batch_size,1))
        print(batch)
    print('epoch done')
    """
    batches_per_epoch = val_gen.n//val_gen.batch_size

    x,y = val_gen.next()
    print(y.shape)
    #for batch in range(batches_per_epoch):
