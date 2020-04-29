import tensorflow as tf


class DenseTranspose(tf.keras.layers.Layer):

    def __init__(self, dense, activation=None, use_bias=False, **kwargs):
        self.dense = dense
        self.use_bias = use_bias
        self.activation = tf.keras.activations.get(activation)
        super().__init__(**kwargs)

    def build(self, batch_input_shape):
        if self.use_bias:
            self.biases = self.add_weight(name='bias',
                                          shape=[self.dense.input_shape[-1]],
                                          initializer='zeros')
        super().build(batch_input_shape)

    def call(self, inputs, **kwargs):
        z = tf.matmul(inputs, self.dense.weights[0], transpose_b=True)
        if self.use_bias:
            z = z + self.biases
        return z

