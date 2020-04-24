from datetime import datetime
from collections.abc import Iterable
import os
import re
import sys

import numpy as np
import tensorflow as tf
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
                 ica_weight=0.005,
                 learning_rate=1e-3,
                 batch_size=256,
                 model_dir=None):

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.nonlinearity = nonlinearity
        self.ica_weight = ica_weight
        self.learning_rate = learning_rate
        self.batch_size = batch_size

        self.model = self.build()

        if model_dir:  # Built model architecture and load saved weights from file
            self.model.load_weights(model_dir)

    def build(self):
        inputs = tf.keras.Input(shape=(self.input_size,), name='input')

        if not self.hidden_size:  # linear RICA, tied weights
            encoder = tf.keras.layers.Dense(self.latent_size,
                                          activation='linear',
                                          use_bias=False,
                                          name='encoder')
            latents = encoder(inputs)
            decoder = layers.DenseTranspose(encoder,
                                            activation='linear',
                                            use_bias=False,
                                            name='decoder')
            reconstructed = decoder(latents)
        else:  # nonlinear version, with hidden layers, not tying weights
            h_encoded = inputs
            if isinstance(self.hidden_size, Iterable):
                h_list = list(self.hidden_size)
            else:
                h_list = [self.hidden_size]

            for idx, h in enumerate(h_list):
                h_encoded = tf.keras.layers.Dense(h, activation=self.nonlinearity,
                                                  name='encoder_'+str(idx))(h_encoded)

            latents = tf.keras.layers.Dense(self.latent_size, activation='linear', name='latent')(h_encoded)
            h_encoded = latents
            for idx, h in enumerate(reversed(h_list)):
                h_encoded = tf.keras.layers.Dense(h, activation=self.nonlinearity,
                                                  name='decoder_'+str(idx))(h_encoded)

            reconstructed = tf.keras.layers.Dense(self.input_size, activation='linear', name='output')(h_encoded)

        model = tf.keras.Model(inputs=inputs, outputs=reconstructed, name='ae')
        sq = tf.multiply(latents, latents)
        kurtosis = tf.reduce_mean(tf.multiply(sq, sq), axis=0)
        loss_ica = tf.reduce_mean(kurtosis)*self.ica_weight
        model.add_loss(loss_ica)
        optimizer = tf.keras.optimizers.Adam(lr=self.learning_rate)
        model.compile(optimizer, loss=tf.keras.losses.mean_squared_error)

        return model

    def train(self, x_train, epochs=10, save=True, model_dir='./model', log_dir='./log'):
        self.model.fit(x_train, x_train,
                       batch_size=self.batch_size, shuffle=True,
                       epochs=epochs)
        if save:
            self.model.save_weights(model_dir)

    def reconstructed(self, inputs):
        return self.model(inputs)

    def inference(self, x):
        latents_out = self.model.get_layer("latent").output
        inference_model = tf.keras.Model(inputs=self.model.input, outputs=latents_out)
        return inference_model.predict(x)
