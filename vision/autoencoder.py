import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, LeakyReLU, Flatten, Dense, Reshape
from tensorflow.keras import layers

class Autoencoder(Model):
  def __init__(self, latent_dim, input_shape):
    super(Autoencoder, self).__init__()
    self.latent_dim = latent_dim   
    self.encoder = tf.keras.Sequential([
        Input(shape=input_shape),
        Conv2D(32, (3, 3), activation='relu', strides=2, padding='same'),
        Conv2D(64, (3, 3), activation='relu', strides=2, padding='same'),
        layers.Flatten(),
        layers.Dense(latent_dim, activation='relu'),
    ])
    self.decoder = tf.keras.Sequential([
        Dense(16 * 16 * 256, activation='relu'),
        layers.Reshape((16, 16, 256)),
        layers.Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same', activation='relu'),
        layers.Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same', activation='relu'),
        Conv2DTranspose(3, (3, 3), activation='sigmoid', padding='same')
    ])

  def call(self, x):
    encoded = self.encoder(x)
    decoded = self.decoder(encoded)
    return decoded