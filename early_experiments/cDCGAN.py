from numpy import zeros
from numpy import ones
from numpy import concatenate
from numpy.random import randn
from numpy.random import randint
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.models import load_model
from util import *
import time
import os
import pickle
from numpy import asarray
from tensorflow.keras.models import load_model
from matplotlib import pyplot

model_name = 'cDCGAN_exp'
BASE_DIR = 'history/{}'.format(model_name)
EVALUATION_DIR = '{}/evaluation'.format(BASE_DIR)
PLOT_DIR = '{}/plots'.format(BASE_DIR)
CHECKPOINT_DIR = '{}/training_checkpoints'.format(BASE_DIR)

if not os.path.exists(BASE_DIR):
	os.makedirs(BASE_DIR)
if not os.path.exists(EVALUATION_DIR):
	os.makedirs(EVALUATION_DIR)
if not os.path.exists(PLOT_DIR):
	os.makedirs(PLOT_DIR)
if not os.path.exists(CHECKPOINT_DIR):
	os.makedirs(CHECKPOINT_DIR)

# define the standalone discriminator model
def define_discriminator(in_shape=(256,256,3), n_classes=2):
	# label input
	in_label = Input(shape=(1,))
	# embedding for categorical input
	li = Embedding(n_classes, 50)(in_label)
	# scale up to image dimensions with linear activation
	n_nodes = in_shape[0] * in_shape[1]
	li = Dense(n_nodes)(li)
	# reshape to additional channel
	li = Reshape((in_shape[0], in_shape[1], 1))(li)
	# image input
	in_image = Input(shape=in_shape)
	# concat label as a channel
	merge = Concatenate()([in_image, li])
	# downsample
	fe = Conv2D(128, (3,3), strides=(2,2), padding='same')(merge)
	fe = LeakyReLU(alpha=0.2)(fe)
	# downsample
	fe = Conv2D(256, (3,3), strides=(2,2), padding='same')(fe)
	fe = LeakyReLU(alpha=0.2)(fe)
	# downsample
	fe = Conv2D(512, (3,3), strides=(2,2), padding='same')(fe)
	fe = LeakyReLU(alpha=0.2)(fe)
	# flatten feature maps
	fe = Flatten()(fe)
	# dropout
	fe = Dropout(0.4)(fe)
	# output
	out_layer = Dense(1, activation='sigmoid')(fe)
	# define model
	model = Model([in_image, in_label], out_layer)
	# compile model
	opt = Adam(lr=0.0002, beta_1=0.5)
	model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
	return model
	
# define the standalone generator model
def define_generator(latent_dim, n_classes=2):
	# label input
	in_label = Input(shape=(1,))
	# embedding for categorical input
	li = Embedding(n_classes, 50)(in_label)
	# linear multiplication
	n_nodes = 16 * 16
	li = Dense(n_nodes)(li)
	# reshape to additional channel
	li = Reshape((16, 16, 1))(li)
	# image generator input
	in_lat = Input(shape=(latent_dim,))
	# foundation for 7x7 image
	n_nodes = 512 * 16 * 16
	gen = Dense(n_nodes)(in_lat)
	gen = LeakyReLU(alpha=0.2)(gen)
	gen = Reshape((16, 16, 512))(gen)
	# merge image gen and label input
	merge = Concatenate()([gen, li])
	# upsample
	print(merge.shape)
	gen = Conv2DTranspose(512, (5,5), strides=(2,2), padding='same')(merge)
	gen = LeakyReLU(alpha=0.2)(gen)
	# upsample
	gen = Conv2DTranspose(256, (5,5), strides=(2,2), padding='same')(gen)
	gen = LeakyReLU(alpha=0.2)(gen)
	print(gen.shape)
	# upsample
	gen = Conv2DTranspose(128, (5,5), strides=(2,2), padding='same')(gen)
	gen = LeakyReLU(alpha=0.2)(gen)
	print(gen.shape)
	# upsample
	out_layer = Conv2DTranspose(3, (5,5), strides=(2,2), padding='same', activation="tanh")(gen)
	print(out_layer.shape)
	# define model
	model = Model([in_lat, in_label], out_layer)
	return model
 
# define the combined generator and discriminator model, for updating the generator
def define_gan(g_model, d_model):
	# make weights in the discriminator not trainable
	d_model.trainable = False
	# get noise and label inputs from generator model
	gen_noise, gen_label = g_model.input
	# get image output from the generator model
	gen_output = g_model.output
	# connect image output and label input from generator as inputs to discriminator
	gan_output = d_model([gen_output, gen_label])
	# define gan model as taking noise and label and outputting a classification
	model = Model([gen_noise, gen_label], gan_output)
	# compile model
	opt = Adam(lr=0.0002, beta_1=0.5)
	model.compile(loss='binary_crossentropy', optimizer=opt)
	return model

 
# create and save a plot of generated images
def save_plot(examples, n, filename):
	# plot images
	for i in range(n * n):
		# define subplot
		pyplot.subplot(n, n, 1 + i)
		# turn off axis
		pyplot.axis('off')
		# plot raw pixel data
		pyplot.imshow(examples[i, :, :, :])
	pyplot.savefig(filename)


# train the generator and discriminator
def train(g_model, d_model, gan_model, img_gen, latent_dim, n_epochs=100, n_classes=2):
	bat_per_epo = img_gen.n//img_gen.batch_size
	half_batch = img_gen.batch_size // 2
	previous = None

	h_d_loss_r = []
	h_d_loss_f = []
	h_g_loss = []
	h_d_a_r = []
	h_d_a_f = []
	# manually enumerate epochs
	for i in range(n_epochs):
		start = time.time()
		# enumerate batches over the training set
		for j in range(bat_per_epo):
		# get randomly selected 'real' samples
			[X_real, labels_real], y_real = generate_real_samples(img_gen, half_batch, previous)
			# update discriminator model weights
			d_loss1, a1 = d_model.train_on_batch([X_real, labels_real], y_real)
			# generate 'fake' examples
			[X_fake, labels], y_fake = generate_fake_samples(g_model, latent_dim, half_batch)
			# update discriminator model weights
			d_loss2, a2 = d_model.train_on_batch([X_fake, labels], y_fake)
			# prepare points in latent space as input for the generator
			[z_input, labels_input] = generate_latent_points(latent_dim, img_gen.batch_size)
			# create inverted labels for the fake samples
			y_gan = ones((img_gen.batch_size, 1))
			# update the generator via the discriminator's error
			g_loss = gan_model.train_on_batch([z_input, labels_input], y_gan)
			# summarize loss on this batch
			print('>%d, %d/%d, d1=%.3f, d2=%.3f g=%.3f' 
			% (i+1, j+1, bat_per_epo, d_loss1, d_loss2, g_loss))
			previous = [X_real, labels_real]
			h_d_loss_r.append(d_loss1)
			h_d_loss_f.append(d_loss2)
			h_g_loss.append(g_loss)
			h_d_a_r.append(a1)
			h_d_a_f.append(a2)
		# Save the model every epoch
		if (i + 1) % 1 == 0:
			g_model.save('{}/generator-e{}.h5'.format(CHECKPOINT_DIR,i+1))
			d_model.save('{}/discriminator-e{}.h5'.format(CHECKPOINT_DIR,i+1))
			for k in range(n_classes):
				if not os.path.exists('{}/epoch{}'.format(PLOT_DIR,i+1)):
					os.makedirs('{}/epoch{}'.format(PLOT_DIR,i+1))
				filename = '{}/epoch{}/class_{}.png'.format(PLOT_DIR,i+1,k)
				latent_points, labels = generate_latent_points(100, 16)
				labels = asarray([k for _ in range(16)])
				X = g_model.predict([latent_points, labels])
				X = (X + 1) / 2.0
				save_plot(X, 4, filename)
		print ('Time for epoch {} is {} sec'.format(i + 1, time.time()-start))
	return h_d_loss_r, h_d_loss_f, h_g_loss, h_d_a_r, h_d_a_f
 
if __name__=="__main__":
	# load image data
	gt = './isic2019/ISIC_2019_Training_GroundTruth_Original.csv'
	images = './isic2019/ISIC_2019_Training_Input'
	cross_val_gens, test_gen = load_isic_2_classes_generators_KFold(images, gt, batch_sz=32, K=5)
	# size of the latent space
	latent_dim = 100
	fold = 1
	metrics = {}
	fold_metrics = {"d_losses_r": None,
					"d_losses_f": None,
					"g_losses": None,
					"d_accuracies_r": None,
					"d_accuracies_f": None,
					"evaluation": None}
	for pair in cross_val_gens:
		print("On Fold ", fold)
		train_gen, val_gen = pair
		# create the discriminator
		d_model = define_discriminator()
		# create the generator
		g_model = define_generator(latent_dim)
		# create the gan
		gan_model = define_gan(g_model, d_model)
		# train model
		h_d_loss_r, h_d_loss_f, h_g_loss, h_d_a_r, h_d_a_f = train(g_model, d_model, gan_model, train_gen, latent_dim, n_epochs=100)
		fold_metrics["d_losses_r"] = h_d_loss_r
		fold_metrics["d_losses_f"] = h_d_loss_f
		fold_metrics["g_losses"] = h_g_loss
		fold_metrics["d_accuracies_r"] = h_d_a_r
		fold_metrics["d_accuracies_f"] = h_d_a_f
		# evaluate model on validation set
		print("Evaluating on val set. Discarding model and saving evaluation score(s)")
		e = evaluate_dicriminator_on_real_images(d_model, val_gen)
		fold_metrics["evaluation"] = e
		metrics[fold] = fold_metrics
		fold += 1

	with open("{}/metrics.p".format(EVALUATION_DIR), "wb") as fp:
		pickle.dump(metrics, fp)