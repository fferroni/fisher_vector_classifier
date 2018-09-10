from functools import partial

import tensorflow as tf
from tensorflow.keras.layers import Input, Conv3D, AvgPool3D, Concatenate, Reshape, Permute, MaxPool3D, Flatten, Dense
from tensorflow.keras.layers import BatchNormalization, Dropout

from fisher_vectors_3d import Modified3DFisherVectors


def inception_block(x, nb_filters=64, name="block1"):

    conv3d = partial(Conv3D, activation="relu", bias=False, padding="same")
    batchn = partial(BatchNormalization, momentum=0.9)

    conv_1x1 = conv3d(nb_filters,
                      [1, 1, 1],
                      name=name+"_conv3d_1x1")(x)
    conv_1x1 = batchn(name=name+"_1x1_bn")(conv_1x1)

    conv_3x3 = conv3d(nb_filters // 2,
                      [3, 3, 3],
                      name=name + "_conv3d_3x3")(conv_1x1)
    conv_3x3 = batchn(name=name + "_3x3_bn")(conv_3x3)

    conv_5x5 = conv3d(nb_filters // 2,
                      [5, 5, 5],
                      name=name + "_conv3d_5x5")(conv_1x1)
    conv_5x5 = batchn(name=name + "_5x5_bn")(conv_5x5)

    avgpool = AvgPool3D(strides=(1, 1, 1), pool_size=(3, 3, 3),
                        padding="same", name=name+"_avgpool")(x)
    avgpool = conv3d(nb_filters,
                     [1, 1, 1],
                     name=name+"_conv3d_avgpool")(avgpool)
    avgpool = batchn(name=name + "_avgpool_bn")(avgpool)

    return Concatenate(axis=-1, name=name+"_concat")([conv_1x1, conv_3x3, conv_5x5, avgpool])


def build_classification_network(batch_size, nb_points, subdivisions, variance):

    points = Input(batch_shape=(batch_size, nb_points, 3), name="points")

    fv = Modified3DFisherVectors(subdivisions, variance, flatten=False)(points)

    fv = Reshape((-1,) + subdivisions, name="reshape_3dmFV")(fv)

    x = Permute((2, 3, 4, 1), name="permute_3dmFV")(fv)

    # convolve
    x = inception_block(x, nb_filters=64, name="block1")
    x = inception_block(x, nb_filters=128, name="block2")
    x = inception_block(x, nb_filters=256, name="block3")
    x = MaxPool3D(name="block3_maxpool")(x)

    x = inception_block(x, nb_filters=256, name="block4")
    x = inception_block(x, nb_filters=512, name="block5")
    x = MaxPool3D(name="block5_maxpool")(x)

    x = Flatten(name="flatten")(x)
    x = Dense(1024, name="fc1", activation="relu")(x)
    x = Dropout(0.3, name="dp1")(x)
    x = Dense(256, name="fc2", activation="relu")(x)
    x = Dropout(0.3, name="dp2")(x)
    x = Dense(128, name="fc2", activation="relu")(x)
    x = Dropout(0.3, name="dp2")(x)
    x = Dense(40, name="output", activation="softmax")(x)

    return tf.keras.models.Model(points, x)
