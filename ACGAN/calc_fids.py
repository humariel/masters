import tensorflow as tf 
import pickle
import os 
import numpy as np
import random

from scipy.stats import truncnorm
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.datasets import cifar10
from skimage.transform import resize
from scipy.linalg import sqrtm
# modules
from Ops.spectral_normalization import SpectralConv2D, SpectralDense
from Ops.ops import ResnetBlock, ResnetBlockUp, ResnetBlockDown
from Ops.attention import Attention
from Ops.global_sum_pooling import GlobalSumPooling2D
from Ops.conditional_batch_normalization import ConditionalBatchNormalization

import argparse

# scale an array of images to a new size
def scale_images(images, new_shape):
    images_list = list()
    for image in images:
        # resize with nearest neighbor interpolation
        new_image = resize(image, new_shape, 0)
        # store
        images_list.append(new_image)
    return np.asarray(images_list)

# generate points in latent space as input for the generator
def generate_latent_points(n_samples, latent_dim=100, n_classes=10, bigGAN=False):
    if not bigGAN:
        # generate points in the latent space
        z_input = np.random.randn(n_samples * latent_dim)
        # reshape into a batch of inputs for the network
        z_input = z_input.reshape(n_samples, latent_dim)
    else:
        z_input = truncnorm.rvs(-2, 2, size=(n_samples, latent_dim), random_state=None).astype(np.float32)
        z_input = z_input * 2.0 # 2.0 is truncation value
    # generate labels
    labels = np.random.randint(0, n_classes, n_samples)
    return [z_input, labels]

def main(model_name):
    inception = InceptionV3(include_top=False, pooling='avg', input_shape=(299,299,3))

    (real_images, _), (_, _) = cifar10.load_data()
    real_images = real_images[:10000]
    np.random.shuffle(real_images)
    # convert integer to floating point values
    real_images = real_images.astype('float32')
    # resize images
    real_images = scale_images(real_images, (299,299,3))
    # pre-process images
    real_images = preprocess_input(real_images)
    # calculate activations
    act1 = inception.predict(real_images)

    custom_objects={'SpectralDense': SpectralDense, 'SpectralConv2D': SpectralConv2D}
    fids = []
    for i in range(250):
        generator = tf.keras.models.load_model('history/bigacgan/{}/training_checkpoints/generator-e{}.h5'.format(model_name, i+1), custom_objects=custom_objects)
        #generate latent points for 50k images
        [z_input, labels] = generate_latent_points(10000, latent_dim=128, bigGAN=True)
        fake_images = generator.predict([z_input, labels])
        fake_images = (fake_images*127.5) + 127.5
        # convert integer to floating point values
        fake_images = fake_images.astype('float32')
        # resize images
        fake_images = scale_images(fake_images, (299,299,3))
        # pre-process images
        fake_images = preprocess_input(fake_images)
        # calculate activations
        act2 = inception.predict(fake_images)

        # calculate mean and covariance statistics
        mu1, sigma1 = act1.mean(axis=0), np.cov(act1, rowvar=False)
        mu2, sigma2 = act2.mean(axis=0), np.cov(act2, rowvar=False)
        # calculate sum squared difference between means
        ssdiff = np.sum((mu1 - mu2)**2.0)
        # calculate sqrt of product between cov
        covmean = sqrtm(sigma1.dot(sigma2))
        # check and correct imaginary numbers from sqrt
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        # calculate score
        fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
        print(i, fid)

        fids.append(fid)
        with open("history/bigacgan/{}/{}_fids.p".format(model_name, model_name), "wb") as fp:
            pickle.dump(fids, fp)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model_name')
    args = parser.parse_args()
    main(args.model_name)