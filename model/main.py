from utils import *
from tensorflow.keras.layers import Input
import os
import matplotlib.pyplot as plt
import datetime
from sklearn.metrics import mean_squared_error
from tqdm import tqdm
import time
import numpy as np

# =========================== Parameters ===========================

IMG_HEIGHT = 150
IMG_WIDTH = 150
IMG_CHANNELS = 1
OUT_HEIGHT = 50
OUT_WIDTH = 50
BATCH_SIZE = 6


# =========================== Functions =============================


def scheduler(epoch):
    """
    This function is used to change the learning rate as the training progresses.

    :param epoch: an int variable showing the current epoch number.
    :return: the learning corresponding the value of the epoch
    """
    if epoch < 3:

        return 0.001

    elif 3 <= epoch < 10:

        return 0.0001

    else:

        return 0.00001


def configure_for_performance(ds, BATCH_SIZE):
    """
    This function prepares the dataset and builds up data pipeline.
    Shuffling, Batching, and Prefetching is performed to enhance the training speed.
    :param ds: the dataset, which is a tf.data.Dataset
    :return: the modified dataset
    """

    # Depending on the amount of the available RAM you could use either just Batching
    # or all of instructions mentioned below.

    ds = ds.cache()
    ds = ds.shuffle(buffer_size=1000)
    ds = ds.batch(BATCH_SIZE)
    ds = ds.prefetch(buffer_size=AUTOTUNE)
    return ds


# =========================== Model Definition =============================

inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))

features = tf.keras.applications.mobilenet_v2.MobileNetV2(
    input_shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), alpha=1.0, include_top=False, weights='imagenet',
    input_tensor=inputs)

output = Dense(2, activation='linear')(features)


class CustomFit(tf.keras.Model):
    def train_step(self, data):
        x, label = data

        with tf.GradientTape() as tape:
            logits = self(x, training=True)
            c = self.compiled_loss(label, logits, regularization_losses=self.losses)

        training_vars = self.trainable_variables
        gradients = tape.gradient(c, training_vars)

        # Step with optimizer
        self.optimizer.apply_gradients(zip(gradients, training_vars))

        self.compiled_metrics.update_state(label, logits)

        return {m.name: m.result() for m in self.metrics}


model = CustomFit(inputs, output)
model.summary()
model.compile(optimizer=tf.keras.optimizers.Adam(), loss="mse", metrics=["mse"])

# =========================== Data Pipelines =============================

# =====================================================
# Preparing the dataset for training or testing phases
# =====================================================

AUTOTUNE = tf.data.AUTOTUNE

# A switch between train and test phases.

is_train = False

# path to save the model weights and parameters
if is_train:
    checkpoint_path = "baseline_weights/COVIDNET-{epoch:04d}.ckpt"
else:
    checkpoint_path = "baseline_weights/COVIDNET-0020.ckpt"

checkpoint_dir = os.path.dirname(checkpoint_path)

# Path to save the logs according to the time
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

# Reading the numpy files of input images and output masks


if is_train:

    # The first 84000 samples are considered as training dataset
    # The next 8000 samples are used for validation

    """
    Loading the train and validation datasets....
    """

    # =========
    # Callbacks
    # =========

    # Callback for saving model's weights

    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        verbose=1,
        save_weights_only=True,
        save_freq='epoch')

    # Callback to save the logs on tensorboard

    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1, update_freq=200)

    # Callback to adjust the learning rate according to the training epoch

    lr_callback = tf.keras.callbacks.LearningRateScheduler(scheduler)

    # Starting the training on GPU

    with tf.device('/gpu:0'):
        model.save_weights(checkpoint_path.format(epoch=0))
        model.fit(train_ds, epochs=20, verbose=1, validation_data=val_ds,
                  callbacks=[cp_callback, tensorboard_callback, lr_callback])

else:

    # loading the model weights

    model.load_weights(checkpoint_path)

    """
    loading test samples...
    """
