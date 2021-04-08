import tensorflow as tf

from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Lambda
from tensorflow.keras.layers import Dense

from typing import Union


def ConditionalBatchNormalization(
    x: tf.Tensor,
    z: tf.Tensor,
    *,
    epsilon: Union[float, tf.Tensor] = 1.0e-8,
    momentum: Union[float, tf.Tensor] = 0.99,
):
    """
    Builds and calls a cross-replica conditional
    batch normalization layer on an image tensor
    `x` and conditioning vector `z`.
    """

    BatchNorm = BatchNormalization

    gamma = Dense(x.shape[-1], bias_initializer="ones")(z)
    beta = Dense(x.shape[-1], bias_initializer="zeros")(z)
    x = BatchNorm(scale=False, center=False, epsilon=epsilon, momentum=momentum)(x)

    def call(args):
        x, g, b = args
        return x * g[:, None, None] + b[:, None, None]

    def output_shape(input_shapes):
        return input_shapes[0]

    return Lambda(call, output_shape=output_shape)([x, gamma, beta])