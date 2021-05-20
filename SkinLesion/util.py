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
TEST_PERC = 15

def get_mnist(buffer_sz=60000, batch_sz=256):
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

    train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
    train_images = (train_images - 127.5) / 127.5 # Normalize the images to [-1, 1]

    test_images = test_images.reshape(test_images.shape[0], 28, 28, 1).astype('float32')
    test_images = (test_images - 127.5) / 127.5 # Normalize the images to [-1, 1]

    dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(buffer_sz).batch(batch_sz)

    return dataset, test_images, test_labels


def load_isic_dataset(images_path, ground_truth_file, batch_sz=32):
    df = pd.read_csv(ground_truth_file)
    df = sklearn.utils.shuffle(df)
    df['image'] = df.apply(lambda row: row['image'] + '.jpg', axis=1)
    df['label'] = df.apply(lambda row: str(np.argmax(row[1:-1])), axis=1).values

    def _preprocess(x):
        return (x-127.5)/127.5 # Normalize the images to [-1, 1]

    data_gen = tf.keras.preprocessing.image.ImageDataGenerator(preprocessing_function=_preprocess, validation_split=0.2)
    train_gen = data_gen.flow_from_dataframe(df, directory=images_path, x_col='image', y_col='label', class_mode='sparse', batch_size=batch_sz, subset="training")
    test_gen = data_gen.flow_from_dataframe(df, directory=images_path, x_col='image', y_col='label', class_mode='sparse', batch_size=batch_sz, subset="validation")

    return train_gen, test_gen

def load_isic_2_classes_generators(images_path, ground_truth_file, batch_sz=32):
    df = pd.read_csv(ground_truth_file)
    columns = df.columns
    melanomes = [x for x in df.values if x[1] == 1.0]
    nv = [x for x in df.values if x[2] == 1.0]
    nv = nv[:len(melanomes)]
    mel_df = pd.DataFrame(melanomes, columns=columns)
    nv_df = pd.DataFrame(nv, columns=columns)
    df = pd.concat([mel_df, nv_df])
    df['image'] = df.apply(lambda row: row['image'] + '.jpg', axis=1)
    df['label'] = df.apply(lambda row: str(np.argmax(row[1:-1])), axis=1).values
    df = sklearn.utils.shuffle(df)

    #split it to test and train dataframe
    train_test_df = np.split(df, [ int(df.shape[0] - (df.shape[0]*(TEST_PERC/100))) ])

    def _preprocess(x):
        return (x-127.5)/127.5 # Normalize the images to [-1, 1]

    data_gen = tf.keras.preprocessing.image.ImageDataGenerator(preprocessing_function=_preprocess)
    train_gen = data_gen.flow_from_dataframe(train_test_df[0], directory=images_path, x_col='image', y_col='label', class_mode='sparse', batch_size=batch_sz)
    test_gen = data_gen.flow_from_dataframe(train_test_df[1], directory=images_path, x_col='image', y_col='label', class_mode='sparse', batch_size=batch_sz)
    return train_gen, test_gen


def load_isic_2_classes_generators_KFold(images_path, ground_truth_file, batch_sz=32, K=5):
    df = pd.read_csv(ground_truth_file)
    columns = df.columns
    melanomes = [x for x in df.values if x[1] == 1.0]
    nv = [x for x in df.values if x[2] == 1.0]
    nv = nv[:len(melanomes)]
    mel_df = pd.DataFrame(melanomes, columns=columns)
    nv_df = pd.DataFrame(nv, columns=columns)
    df = pd.concat([mel_df, nv_df])
    df['image'] = df.apply(lambda row: row['image'] + '.jpg', axis=1)
    df['label'] = df.apply(lambda row: str(np.argmax(row[1:-1])), axis=1).values
    df = sklearn.utils.shuffle(df)

    #split it to test and train dataframe
    train_test_df = np.split(df, [ int(df.shape[0] - (df.shape[0]*(TEST_PERC/100))) ])

    # now split train df into K folds
    kfolds = np.array_split(df, K)

    def _preprocess(x):
        return (x-127.5)/127.5 # Normalize the images to [-1, 1]
    
    cross_val_gens = []
    for i in range(K):
        train_dfs = []
        for j in range(K):
            if i != j:
                train_dfs.append(kfolds[j])
        train_df = pd.concat(train_dfs)
        gen = tf.keras.preprocessing.image.ImageDataGenerator(preprocessing_function=_preprocess)
        train_gen = gen.flow_from_dataframe(train_df, directory=images_path, x_col='image', y_col='label', class_mode='sparse', batch_size=batch_sz)
        val_gen = gen.flow_from_dataframe(kfolds[i], directory=images_path, x_col='image', y_col='label', class_mode='sparse', batch_size=batch_sz)
        cross_val_gens.append((train_gen,val_gen))

    test_gen = tf.keras.preprocessing.image.ImageDataGenerator(preprocessing_function=_preprocess)
    test_gen = test_gen.flow_from_dataframe(train_test_df[1], directory=images_path, x_col='image', y_col='label', class_mode='sparse', batch_size=batch_sz)
    return cross_val_gens, test_gen


def load_isic_2_classes_dataset(images_path, ground_truth_file, batch_size=32):
    df = pd.read_csv(ground_truth_file)
    df = sklearn.utils.shuffle(df)
    columns = df.columns
    melanomes = [x for x in df.values if x[1] == 1.0]
    nv = [x for x in df.values if x[2] == 1.0]
    nv = nv[:len(melanomes)]
    mel_df = pd.DataFrame(melanomes, columns=columns)
    nv_df = pd.DataFrame(nv, columns=columns)
    df = pd.concat([mel_df, nv_df])
    df['image'] = df.apply(lambda row: row['image'] + '.jpg', axis=1)
    df['label'] = df.apply(lambda row: np.argmax(row[1:-1]), axis=1).values

    def iterator():
        for image_path,label in zip(df['image'], df['label']):
            image =Image.open(images_path + image_path)
            image =  np.asarray(image.resize((256,256)))
            image = (image-127.5)/127.5
            yield image, np.array([label])

    dataset = tf.data.Dataset.from_generator(
            iterator,
            output_types=(tf.float32,tf.int32),
            output_shapes=(tf.TensorShape([256,256,3]),tf.TensorShape([1]))
        ).batch(batch_size, drop_remainder=True)
    return dataset

##################################################################################
#############################IMAGE GENERATION#####################################
# select real samples
def generate_real_samples(img_gen, n_samples, previous):
	# split into images and labels
	images, labels = img_gen.next()
	#solve last batch of epoch size issue
	if images.shape[0] != img_gen.batch_size:
		diff = img_gen.batch_size - images.shape[0] + 1
		X_prior, labels_prior = previous
		images = concatenate((images, X_prior[-diff:-1]))
		labels = concatenate((labels, labels_prior[-diff:-1]))
	# choose random instances
	ix = randint(0, images.shape[0], n_samples)
	# select images and labels
	X, labels = images[ix], labels[ix]
	# generate class labels
	y = ones((n_samples, 1))
	return [X, labels], y

# generate points in latent space as input for the generator
def generate_latent_points(latent_dim, n_samples, n_classes=2):
	# generate points in the latent space
	x_input = randn(latent_dim * n_samples)
	# reshape into a batch of inputs for the network
	z_input = x_input.reshape(n_samples, latent_dim)
	# generate labels
	labels = randint(0, n_classes, n_samples)
	return [z_input, labels]
 
# use the generator to generate n fake examples, with class labels
def generate_fake_samples(generator, latent_dim, n_samples):
	# generate points in latent space
	z_input, labels_input = generate_latent_points(latent_dim, n_samples)
	# predict outputs
	images = generator.predict([z_input, labels_input])
	# create class labels
	y = zeros((n_samples, 1))
	return [images, labels_input], y
##################################################################################
################################EVALUATION########################################

def evaluate_dicriminator_on_real_images(d_model, generator):
    bat_per_epo = generator.n//generator.batch_size
    previous = None
    metrics = []
    for i in range(bat_per_epo):
        [images, labels], y = generate_real_samples(generator, generator.batch_size, previous)
        y = [int(e[0]) for e in y]
        predictions = d_model.predict([images,labels])
        predictions = [int(round(p[0])) for p in predictions]
        e = sklearn.metrics.classification_report(y, predictions, output_dict=True)
        metrics.append(e)
        previous = [images, labels]
    # return metrics
    return metrics

def evaluate_discriminator_on_fake_images(d_model,g_model, latent_dim):
    # NEEDS CHANGE
    bat_per_epo = 226
    metrics = []
    for i in range(bat_per_epo):
        [images, labels], y = generate_fake_samples(g_model, latent_dim, 32) # NEEDS CHANGE
        y = [int(e[0]) for e in y]
        predictions = d_model.predict([images,labels])
        predictions = [int(round(p[0])) for p in predictions]
        e = sklearn.metrics.classification_report(y, predictions, output_dict=True)
        metrics.append(e)
        previous = [images, labels]
    # return metrics
    return metrics

##################################################################################


if __name__=='__main__':
    images = './isic2019/ISIC_2019_Training_Input/'
    gt = './isic2019/ISIC_2019_Training_GroundTruth_Original.csv'
    t,_ = load_isic_2_classes_generators_KFold(images, gt)
#    for _ in range(2):
#        print('´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´')
#        start = True
#        for batch in range(t.n//t.batch_size):
#            ims = t.next()
#            if start:
#                start = False
#                print(ims[0])