import tensorflow as tf
from tensorflow.keras import layers , models

class TrafficSignNet:
    @staticmethod
    def build(width, height, depth, classes):
        inputs = tf.keras.Input(shape=(height, width, depth))
        x = layers.Conv2D(8,(5,5), padding='same')(inputs)
        x= layers.BatchNormalization(axis= -1)(x)
        x = layers.Activation("relu")(x)
        x = layers.MaxPooling2D(pool_size=(2, 2))(x)

        # First set of (CONV => RELU => BN) => (CONV => RELU => BN) => POOL
        x = layers.Conv2D(16, (3, 3), padding="same")(inputs)
        x = layers.BatchNormalization(axis=-1)(x)
        x = layers.Activation("relu")(x)
        x = layers.Conv2D(16, (3, 3), padding="same")(x)
        x = layers.BatchNormalization(axis=-1)(x)
        x = layers.Activation("relu")(x)
        x = layers.MaxPooling2D(pool_size=(2, 2))(x)

        # Second set of (CONV => RELU => BN) => (CONV => RELU => BN) => POOL
        x = layers.Conv2D(32, (3, 3), padding="same")(x)
        x = layers.BatchNormalization(axis=-1)(x)
        x = layers.Activation("relu")(x)
        x = layers.Conv2D(32, (3, 3), padding="same")(x)
        x = layers.BatchNormalization(axis=-1)(x)
        x = layers.Activation("relu")(x)
        x = layers.MaxPooling2D(pool_size=(2, 2))(x)

        model = models.Model(inputs=inputs, outputs=x)

        return model