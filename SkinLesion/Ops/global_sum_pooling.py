import tensorflow as tf
from tensorflow.keras.layers import Lambda

def GlobalSumPooling2D():
    """
    Layer that sums over all spatial locations,
    preserving batch and channels dimensions.
    """
    def call(x):
        return tf.reduce_sum(x, axis=(1, 2))

    def output_shape(input_shape):
        return input_shape[0], input_shape[-1]

    return Lambda(call, output_shape=output_shape)
