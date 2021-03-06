from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import Dot
from tensorflow.keras.layers import Add
from tensorflow.keras.layers import Conv2D
from tensorflow.keras import backend as K
import tensorflow_addons as tfa

from functools import partial

#from .spectral_normalization import SpectralConv2D


def Attention(x, use_bias=True, epsilon=1.0e-8):
    """
    Constructs a self-attention layer.
    Cf. https://arxiv.org/pdf/1805.08318.pdf,
    section 3; also see the corresponding code:
    https://github.com/brain-research/self-attention-gan
    """
    batch, height, width, channels = K.int_shape(x)
    space = height * width
    f = tfa.layers.SpectralNormalization(Conv2D(channels // 8, 1, use_bias=False))(x)
    f = Reshape((space, channels // 8))(f)
    xbar = AveragePooling2D()(x)
    g = tfa.layers.SpectralNormalization(Conv2D(channels // 8, 1, use_bias=False))(xbar)
    g = Reshape((space // 4, channels // 8))(g)
    h = tfa.layers.SpectralNormalization(Conv2D(channels // 2, 1, use_bias=False))(xbar)
    h = Reshape((space // 4, channels // 2))(h)
    attn = Dot((2, 2))([f, g])
    attn = Activation("softmax", dtype="float32")(attn)
    y = Dot((2, 1))([attn, h])
    y = Reshape((height, width, channels // 2))(y)
    y = tfa.layers.SpectralNormalization(Conv2D(channels, 1, use_bias=use_bias))(y)
    return Add()([x, y])