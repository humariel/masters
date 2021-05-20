import tensorflow as tf
from tensorflow.keras.models import load_model

from scipy.stats import truncnorm

import numpy as np
from PIL import Image

# modules
from Ops.spectral_normalization import SpectralConv2D, SpectralDense
from Ops.ops import ResnetBlock, ResnetBlockUp, ResnetBlockDown
from Ops.attention import Attention
from Ops.global_sum_pooling import GlobalSumPooling2D
from Ops.conditional_batch_normalization import ConditionalBatchNormalization

def generate_latent_points(latent_dim, n_samples, label):
    # generate points in the latent space
    z_input = truncnorm.rvs(-2, 2, size=(n_samples, latent_dim), random_state=None).astype(np.float32)
    z_input = z_input * 2.0 # 2.0 is truncation value
    # generate labels
    labels = np.asarray([label for _ in range(n_samples)])
    return z_input, labels

def save_images(images, base_name):
    count = 0
    for im in images:
        x = Image.fromarray((im*255).astype(np.uint8))
        x.save('{}/seed{:04d}.png'.format(base_name, count))
        count += 1



if __name__ == '__main__':

    model_path = './history/bigacgan/bigacgan-cifar10-2/training_checkpoints/generator-e250.h5'
    dataset_dir = '../Datasets/cifar-10-generated_bigacgan'
    custom_objects={'SpectralDense': SpectralDense, 'SpectralConv2D': SpectralConv2D}
    generator = load_model(model_path, custom_objects=custom_objects)

    #generate images for all classes of cifar10
    for i in range(10):
        z_input, labels = generate_latent_points(128, 5000, i)
        X = generator.predict([z_input, labels])
        X = (X+1) / 2  

        base_name = dataset_dir + '/{}'.format(i)
        save_images(X, base_name)
            