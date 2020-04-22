import tensorflow as tf
import keras


class DenseTranspose(keras.layers.Layer):

    def __init__(self, dense, activation=None, use_bias=False, **kwargs):
        self.dense = dense
        self.use_bias = use_bias
        self.biases = self.add_weight(name='bias',
                                      shape=[self.dense.input_shape[-1]],
                                      initializer='zeros')
        self.activation = keras.activations.get(activation)
        super().__init__(**kwargs)

    def build(self, batch_input_shape):
        super().build(batch_input_shape)

    def call(self, inputs, **kwargs):
        z = tf.matmul(inputs, self.dense.weights[0], transpose_b=True)
        if self.use_bias:
            z = z + self.biases
        return z


class Encoder(keras.layers.Layer):
    def __init__(self,
                 hidden_size,
                 latent_size,
                 activation,
                 name='encoder',
                 **kwargs):
        super(Encoder, self).__init__(name=name, **kwargs)
        self.hidden = keras.layers.Dense(hidden_size, activation=activation)
        self.latent = keras.layers.Dense(latent_size, activation='linear')

    def call(self, inputs, **kwargs):
        x = self.hidden(inputs)
        return self.latent(x)


class Decoder(keras.layers.Layer):
    def __init__(self,
                 hidden_size,
                 output_size,
                 activation,
                 name='decoder',
                 **kwargs):
        super(Decoder, self).__init__(name=name, **kwargs)
        self.hidden = keras.layers.Dense(hidden_size, acitvation=activation)
        self.reconstructed = keras.layers.Dense(output_size, activation='linear')

    def call(self, latents, **kwargs):
        x = self.hidden(latents)
        return self.reconstructed(x)
