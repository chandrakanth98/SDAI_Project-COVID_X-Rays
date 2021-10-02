from tensorflow.keras.layers import Input
import os
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import  Dense
from PIL import Image
import argparse


def pure_pil_alpha_to_color_v2(image, color=(255, 255, 255)):
    """Alpha composite an RGBA Image with a specified color.

    Simpler, faster version than the solutions above.

    Source: http://stackoverflow.com/a/9459208/284318

    Keyword Arguments:
    image -- PIL RGBA Image object
    color -- Tuple r, g, b (default 255, 255, 255)

    """
    image.load()  # needed for split()
    background = Image.new('RGB', image.size, color)
    background.paste(image, mask=image.split()[3])  # 3 is the alpha channel
    return background


def convert(rgba):
    rgb = pure_pil_alpha_to_color_v2(rgba)
    return np.array(rgb)


IMG_HEIGHT = 224
IMG_WIDTH = 224
IMG_CHANNELS = 3
BATCH_SIZE = 10


def backprop_receptive_field(image_dir, weight_path, use_trained=False,
                             use_max_activation=False):
    """
        CNN Receptive Field Computation Using Backprop with TensorFlow
        Adopted from: https://learnopencv.com/cnn-receptive-field-computation-using-backprop-with-tensorflow/
    :param image: input image
    :param directory: directory to save the heatmaps
    :param use_trained: boolean, whether to use trained model or random
    :param checkpoint: path to the trained weights
    """

    # Implementation of Multi-Conv-FCN
    inputs = Input((224, 224, 3))

    mbnet = tf.keras.applications.InceptionV3(input_shape=(224, 224, 3), include_top=False,
                                              weights='imagenet',
                                              input_tensor=inputs)
    for layer in mbnet.layers:
        layer.trainable = False

    features = mbnet(inputs)
    flat_features = tf.keras.layers.Flatten()(features)
    l1 = Dense(4096, 'relu')(flat_features)
    l1 = tf.keras.layers.BatchNormalization()(l1)
    l1 = Dense(4096, 'relu')(l1)
    l1 = tf.keras.layers.BatchNormalization()(l1)
    dp = tf.keras.layers.Dropout(0.4)(l1)
    output = Dense(3)(dp)
    model = tf.keras.Model(inputs, features)

    # Loading the weights
    if use_trained:
        model.load_weights(weight_path)
        print('model weights loaded.')

    new_image = plt.imread(image_dir)
    if len(new_image.shape) == 4:
        new_image = convert(image_dir)
    elif len(new_image.shape) == 2:
        image = plt.imread(image_dir)
        new_image = np.zeros(list(image.shape)[:2] + [3], dtype=np.float32)
        new_image[:, :, 0] = image
        new_image[:, :, 1] = image
        new_image[:, :, 2] = image
    new_image = new_image.astype(np.uint8)
    image = Image.fromarray(new_image).resize((224, 224))
    image = np.array(image, dtype=np.float32) / 255.
    image = tf.expand_dims(image, 0)

    pred = model.predict(image)
    preds = tf.transpose(pred, perm=[0, 3, 1, 2])
    preds = tf.nn.softmax(preds, axis=1)
    # find class with the maximum score in the n × m output map
    pred = tf.math.reduce_max(preds, axis=1)
    class_idx = tf.math.argmax(preds, axis=1)
    row_max = tf.math.reduce_max(pred, axis=1)
    row_idx = tf.math.argmax(pred, axis=1)
    col_idx = tf.math.argmax(row_max, axis=1)
    predicted_class = tf.gather_nd(class_idx, (0, tf.gather_nd(row_idx, (0, col_idx[0])), col_idx[0]))
    score_map = tf.expand_dims(preds[0, predicted_class, :, :], 0).numpy()

    input = tf.ones_like(image)
    out = model.predict(image)

    receptive_field_mask = tf.Variable(tf.zeros_like(out))

    if not use_max_activation:
        receptive_field_mask[:, :, :, predicted_class].assign(score_map)
    else:
        scoremap_max_row_values = tf.math.reduce_max(score_map, axis=1)
        max_row_id = tf.math.argmax(score_map, axis=1)
        max_col_id = tf.math.argmax(scoremap_max_row_values, axis=1).numpy()[0]
        max_row_id = max_row_id[0, max_col_id].numpy()
        # update grad
        receptive_field_mask = tf.tensor_scatter_nd_update(
            receptive_field_mask,
            [(0, max_row_id, max_col_id, 0)], [1],
        )
    grads = []
    with tf.GradientTape() as tf_gradient_tape:
        print('Hi I am here')
        tf_gradient_tape.watch(input)
        # get the predictions
        preds = model(input)
        # apply the mask
        pseudo_loss = preds * receptive_field_mask
        pseudo_loss = K.mean(pseudo_loss)
        # get gradient
        grad = tf_gradient_tape.gradient(pseudo_loss, input)
        grad = tf.transpose(grad, perm=[0, 3, 1, 2])
        grads.append(grad)
    return grads[0][0, 0]


def covid_net(image_path, weight_path):
    class_to_label = {0: 'covid', 1: 'normal', 2: 'pneumonia'}
    inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))

    mbnet = tf.keras.applications.InceptionV3(input_shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), include_top=False,
                                              weights='imagenet',
                                              input_tensor=inputs)
    for layer in mbnet.layers:
        layer.trainable = False

    features = mbnet(inputs)
    flat_features = tf.keras.layers.Flatten()(features)
    l1 = Dense(4096, 'relu')(flat_features)
    l1 = tf.keras.layers.BatchNormalization()(l1)
    l1 = Dense(4096, 'relu')(l1)
    l1 = tf.keras.layers.BatchNormalization()(l1)
    dp = tf.keras.layers.Dropout(0.4)(l1)
    output = Dense(3)(dp)
    model = tf.keras.Model(inputs, output)

    model.load_weights(weight_path)
    new_image = plt.imread(image_path)
    if len(new_image.shape) == 4:
        new_image = convert(image_path)
    elif len(new_image.shape) == 2:
        image = plt.imread(image_path)
        new_image = np.zeros(list(image.shape)[:2] + [3], dtype=np.float32)
        new_image[:, :, 0] = image
        new_image[:, :, 1] = image
        new_image[:, :, 2] = image
    new_image = new_image.astype(np.uint8)
    image = Image.fromarray(new_image).resize((224, 224))
    image = np.array(image, dtype=np.float32) / 255.
    image = tf.expand_dims(image, 0)

    labels = tf.nn.softmax(model.predict(image))
    labels = tf.nn.softmax(labels, axis=1)
    # find class with the maximum score in the n × m output map
    label = tf.math.reduce_max(labels, axis=1)
    class_idx = tf.math.argmax(labels, axis=1)
    return class_to_label[class_idx.numpy()[0]]


if __name__ == '__main__':
    # Reading the execution arguments from command line
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str, required=True)
    parser.add_argument('--save_path', type=str, required=True)
    args = parser.parse_args()
    image_dir = args.image_path
    save_path = args.save_path
    weight_path = 'D:/AI-Project/AI-Backend/Python/baseline_weights/InceptionV3-0004.ckpt'
    result = covid_net(image_dir, weight_path)
    os.makedirs(save_path + '/' + result)

    # In case of positive covid we create a visualization
    if result == 'covid':
        from tensorflow.python.ops.numpy_ops import np_config

        np_config.enable_numpy_behavior()

        grads = backprop_receptive_field(
            image_dir=image_dir,
            weight_path=weight_path,
            use_trained=True,
            use_max_activation=False
        )
        plt.imsave(save_path + '/' + result + '/result.png', grads, cmap='hot')
