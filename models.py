from datetime import datetime
import os
import re
import sys

import numpy as np
import tensorflow as tf
from tensorflow import keras
import layers


class ICANN:
    """
    Independent component analysis with neural network
    Based on the following algorithms and implementations:
    RICA: http://ai.stanford.edu/~quocle/LeKarpenkoNgiamNg.pdf
    ICAE: A_Penalized_Autoencoder_Approach_for_Nonlinear_Independent_Component_Analysis
          https://github.com/TianwenWei/ICAE.git
    """

    def __init__(self,
                 input_size=None,
                 hidden_size=None,
                 latent_size=None,
                 nonlinearity='relu',
                 learning_rate=1e-3,
                 batch_size=256,
                 model_dir=None):

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.nonlinearity = nonlinearity
        self.learning_rate = learning_rate
        self.batch_size = batch_size

        if model_dir:
            pass  # under construction: update architecture from model name, load saved weights to built model

        self.model = self.build()

    def build(self):
        inputs = keras.Input(shape=(self.input_size,), name='encoder_input')

        if not self.hidden_size:  # linear RICA
            encoder = keras.layers.Dense(self.latent_size,
                                         activation='linear',
                                         use_bias=False,
                                         name='encoder')
            decoder = layers.DenseTranspose(encoder,
                                            activation='linear',
                                            use_bias=False,
                                            name='decoder')
            latents = encoder(inputs)
            reconstructed = decoder(latents, use_bias=False)
        else:  # nonlinear version, with one hidden layer
            encoder = layers.Encoder(hidden_size=self.hidden_size,
                                     latent_size=self.latent_size,
                                     activation=self.nonlinearity,
                                     name='encoder')
            decoder = layers.Decoder(hidden_size=self.hidden_size,
                                     output_size=self.input_size,
                                     activation=self.nonlinearity,
                                     name='decoder')
            latents = encoder(inputs)
            reconstructed = decoder(latents)

        model = keras.Model(inputs=inputs, outputs=reconstructed, name='ae')
        sq = tf.multiply(latents, latents)
        kurtosis = tf.reduce_mean(tf.multiply(sq, sq), axis=0)
        loss_ica = tf.reduce_mean(kurtosis)
        model.add_loss(loss_ica)
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        model.compile(optimizer, loss=tf.keras.losses.mean_squared_error())

        return model

    def train(self, x_train, epochs=10, save=True, model_dir='./model', log_dir='./log'):
        self.model.fit(x_train, x_train,
                       batch_size=self.batch_size,
                       epochs=epochs)
        if save:
            self.model.save_weights(model_dir)

    def reconstructed(self, inputs):
        return self.model(inputs)

    def inference(self, inputs):
        latents_out = self.model.get_layer("encoder").output
        inference_model = keras.Model(inputs=self.model.input, outputs=latents_out)
        return inference_model(inputs)
