import cv2
import os
import numpy as np
import random
import warnings
import tensorflow as tf
print(tf.__version__)
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, LeakyReLU, Flatten, Dense, Reshape
from tensorflow.keras import layers
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from skimage.metrics import structural_similarity as ssim

import matplotlib.pyplot as plt
from autoencoder import Autoencoder

def load_data(data_dir, image_size=(64, 64, 3)):
    image_names = os.listdir(data_dir)
    image_names = [x for x in image_names if x.endswith('.jpg') or x.endswith('.png') or x.endswith('.jpeg')]
    train_data = []
    for image_name in image_names:
        image = cv2.imread(os.path.join(data_dir, image_name))
        image = image / 255.0
        train_data.append(image)
    return train_data

def load_data_noise(data_dir, noise_mean=0, noise_std=0.01):
    image_names = os.listdir(data_dir)
    image_names = [x for x in image_names if x.endswith('.jpg') or x.endswith('.png') or x.endswith('.jpeg')]
    train_data = []
    for image_name in image_names:
        image = cv2.imread(os.path.join(data_dir, image_name), 0)
        image = cv2.resize(image, (64, 64, 3)) / 255.0
        noise = np.random.normal(noise_mean, noise_std, size=image.shape)
        noisy_image = np.clip(image + noise, 0, 1)
        train_data.append(noisy_image)
    return train_data

def create_autoencoder_new(latent_dim, input_shape=(64, 64, 3)):
    # Encoder
    encoder_inputs = Input(shape=input_shape)
    x = Conv2D(32, (3, 3), activation='relu', strides=2, padding='same')(encoder_inputs)
    x = Conv2D(64, (3, 3), activation='relu', strides=2, padding='same')(x)
    x = Flatten()(x)
    encoded = Dense(latent_dim)(x)

    # Decoder
    x = Dense(16 * 16 * 256, activation='relu')(encoded)
    x = layers.Reshape((16, 16, 256))(x)
    x = layers.Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same', activation='relu')(x)
    x = layers.Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same', activation='relu')(x)

    # Output layer
    decoded = Conv2DTranspose(3, (3, 3), activation='sigmoid', padding='same')(x)

    # Autoencoder
    autoencoder = tf.keras.Model(encoder_inputs, decoded, name='autoencoder')
    return autoencoder
    

def dssim_loss(y_true, y_pred):
    return 1/2 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, 1.0))/2

def model_name(dataset_name, latent_dim, training_loss, batch_size):
    return f'{dataset_name}_dim_{latent_dim}_loss_{training_loss}_batch_{batch_size}.hdf5'

def train_model(dataset_name, model_basedir, input_shape=(64, 64, 3), latent_dim=100, training_loss='ssim', random_crop=False, batch_size=8, epochs=20):
    #
    # Read data
    #

    # X_train = load_data("train/encoder/cropped")
    X_train = load_data("images/", input_shape)
    X_train = np.array(X_train)
    #X_train = np.expand_dims(X_train, axis=-1)

    Y_train = X_train
    X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.2, shuffle=True)
    print(f'Loaded data: train {X_train.shape}, validation {Y_val.shape}')

    #
    # Create autoencoder
    #
    autoencoder = Autoencoder(input_shape=input_shape, latent_dim=latent_dim)

    #
    # Set training loss function and optimizer
    #
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=1e-3,
        decay_steps=10000,
        decay_rate=0.9)

    opt = Adam(learning_rate=lr_schedule)

    if training_loss == 'mse':
        autoencoder.compile(loss='mse', optimizer=opt)
    elif training_loss == 'ssim':
        autoencoder.compile(loss=dssim_loss, optimizer=opt)

    #    
    # Set callbacks
    #
    model_filename = model_name(dataset_name, latent_dim, training_loss, batch_size)
    path_to_save_model = f'{model_basedir}/{model_filename}'

    model_checkpoint_callback = ModelCheckpoint(
      save_weights_only=False, filepath=path_to_save_model,
      monitor='val_loss', save_best_only=True)
    early_stopping_callback = EarlyStopping(
      monitor='val_loss', min_delta=0, patience=50, verbose=0, mode='auto')

    #
    # Training
    #
    autoencoder.fit(
      x=X_train, y=Y_train,
      epochs=epochs, batch_size=batch_size,
      shuffle=True, validation_data=(X_val, Y_val),
      callbacks=[])
    
    return autoencoder

def load_model(dataset_name, model_basedir, input_shape=(64, 64, 3), latent_dim=100, training_loss='ssim', batch_size=8):
    autoencoder = create_autoencoder_new(input_shape=input_shape, latent_dim=latent_dim)
    
    model_filename = "6_test.hdf5"
    if model_filename in os.listdir(f'{model_basedir}'):
        autoencoder.load_weights(f'{model_basedir}/{model_filename}')
    else:
        raise FileNotFoundError(f'Model {model_filename} not found')
        
    return autoencoder

dataset_name = 'grid'
latent_dim = 50
batch_size = 8
training_loss = 'ssim'
# load_model = False
random_crop = True

model_basedir = 'models'

autoencoder = train_model( 
    dataset_name = dataset_name,
    model_basedir = model_basedir,
    input_shape=(64, 64, 3),
    latent_dim = latent_dim,
    training_loss = training_loss,
    random_crop = random_crop,
    batch_size = batch_size,
    epochs =80)

autoencoder.encoder.save_weights("models/encoder.h5")