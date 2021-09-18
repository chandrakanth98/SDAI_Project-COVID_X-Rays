import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, concatenate, Conv2DTranspose, BatchNormalization, MaxPooling2D, Dropout, \
    ReLU, Reshape, Flatten, Dense
from tensorflow.keras import initializers
import numpy as np


class ConvBlock(tf.keras.layers.Layer):
    """
            Definition of Convolutional Block, which applies a (kernel,kernel) convolution followed
            by batch-norm, LeakyReLU, and dropout.

            Attributes
            ----------
            depth : int
                Depth of the layer
            kernel: int
                Kernel size for the convolution
            dropout: float
                Float between 0 and 1. Fraction of the input units to drop.
            name: string
                Name of the layer

    """

    def __init__(self, depth, kernel, dropout=0.4, name='ConvBlock'):
        super(ConvBlock, self).__init__()

        self.conv1 = Conv2D(depth, (kernel, kernel), activation='linear', kernel_initializer='he_normal',
                            name=name + '_conv1')

        self.batch1 = BatchNormalization(name=name + '_batch1')

        self.relu1 = ReLU()

        self.dropout = Dropout(dropout, name=name + '_dropout')

    def call(self, input):
        conv1 = self.conv1(input)
        batch1 = self.batch1(conv1)
        relu1 = self.relu1(batch1)
        dropout = self.dropout(relu1)

        return dropout


class ChannelAttention(tf.keras.layers.Layer):
    """
       Implementation of channel attention according to https://arxiv.org/pdf/1807.06521.pdf

    Attributes
    ----------
    channel : int
           The number of channels in the input feature map

    ratio: int
           The factor by which the information from all the channels are shrank with.

    name: string
           Name of the layer

    """

    def __init__(self, channel, ratio=8, name='ChannelAttention'):
        super(ChannelAttention, self).__init__()

        # The shared MLP layers to extract the importance of channels

        self.mlp1 = tf.keras.layers.Dense(units=channel // ratio, activation=tf.nn.relu, name=name + '_mlp0',
                                          kernel_regularizer=tf.keras.regularizers.l2(l2=0.001))

        self.mlp2 = tf.keras.layers.Dense(units=channel, activation=tf.nn.relu, name=name + '_mlp1',
                                          kernel_regularizer=tf.keras.regularizers.l2(l2=0.001))

        self.concat = concatenate

    def call(self, input):
        # Performing average pooling on a N*H*W*C map to obtain a N*1*1*C map
        avg_pool = tf.reduce_mean(input, axis=[1, 2], keepdims=True)

        avg_pool = self.mlp1(avg_pool)
        avg_pool = self.mlp2(avg_pool)

        # Performing max pooling on a N*H*W*C map to obtain a N*1*1*C map
        max_pool = tf.reduce_max(input, axis=[1, 2], keepdims=True)
        max_pool = self.mlp1(max_pool)
        max_pool = self.mlp2(max_pool)

        scale = tf.sigmoid(avg_pool + max_pool, 'sigmoid')

        return input * scale


class SpatialAttention(tf.keras.layers.Layer):
    """
       Implementation of spatial attention according to https://arxiv.org/pdf/1807.06521.pdf

    Attributes
    ----------
    kernel : int
           kernel size for the convolution operation

    name: string
           Name of the layer

    """

    def __init__(self, kernel, name='SpatialAttention'):
        super(SpatialAttention, self).__init__()

        self.conv = Conv2D(1, (kernel, kernel), kernel_initializer='he_normal',
                           padding='same', name=name + '_conv', kernel_regularizer=tf.keras.regularizers.l2(l2=0.001))
        self.concat = concatenate

    def call(self, input):

        # Performing average and max pooling on the channels to obtain a N*H*W*1 from a N*H*W*C feature map.
        avg_pool = tf.reduce_mean(input, axis=[3], keepdims=True)
        max_pool = tf.reduce_max(input, axis=[3], keepdims=True)

        concat = self.concat([avg_pool, max_pool], axis=-1)

        conv = tf.sigmoid(self.conv(concat))

        return input * tf.repeat(conv, [input.shape[-1]], axis=-1)


class CBAM(tf.keras.layers.Layer):
    """
       Implementation of  Convolutional Block Attention Module attention (CBAM)
       according to https://arxiv.org/pdf/1807.06521.pdf

    Attributes
    ----------
    kernel : int
           Kernel size for the convolution operation
    channel : int
           The number of channels of the input feature map
    ratio:
           The factor by which the information from all the channels are shrank with.
    name: string
           Name of the layer

    """
    def __init__(self, channel, kernel, ratio=8, name='CBAM'):
        super(CBAM, self).__init__()

        self.channel_attention = ChannelAttention(channel, ratio, name=name + 'ChannelAttention')

        self.spatial_attention = SpatialAttention(kernel, name=name + 'SpatialAttention')

    def call(self, input):

        ca = self.channel_attention(input)

        sa = self.spatial_attention(ca)

        return sa + input

