# https://keras.io/examples/vision/image_classification_efficientnet_fine_tuning/
import os
from datetime import datetime

import numpy as np
import tensorflow as tf
from tensorflow.keras import callbacks, optimizers
from tensorflow.keras import layers
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras.models import Sequential

# IMG_SIZE is determined by EfficientNet model choice
IMG_SIZE = (224, 224)
BATCH_SIZE = 64
BUFFER_SIZE = 32
NUM_CLASSES = 3
DATASET_PATH = "./datasets/GT_cut/"
CKPT_PATH = "./efficientnet-b0/efficientnetb0_notop.h5"
TRAIN_PATH = "./training/efficientnet-b0_1/"
CHECKPOINT_PATH = "checkpoints/cp.ckpt"

logdir = TRAIN_PATH + "logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = callbacks.TensorBoard(log_dir=logdir)

# Create a callback that saves the model's weights
checkpoint_dir = os.path.dirname(TRAIN_PATH + CHECKPOINT_PATH)
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=CHECKPOINT_PATH, save_weights_only=True, verbose=1, save_best_only=True, monitor='val_accuracy', mode='max')

strategy = tf.distribute.MirroredStrategy()

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    DATASET_PATH + "train/",
    label_mode="categorical",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    DATASET_PATH + "val/",
    label_mode="categorical",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

train_ds = train_ds.prefetch(tf.data.experimental.AUTOTUNE)

# Augmentation
img_augmentation = Sequential(
    [
        preprocessing.RandomRotation(factor=0.15),
        preprocessing.RandomTranslation(height_factor=0.1, width_factor=0.1),
        preprocessing.RandomFlip(),
        preprocessing.RandomContrast(factor=0.1),
    ],
    name="img_augmentation",
)


# Transfer learning
def build_model(num_classes):
    inputs = layers.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
    x = img_augmentation(inputs)
    model = EfficientNetB0(include_top=False, input_tensor=x, weights=CKPT_PATH)

    # Freeze the pretrained weights
    model.trainable = False

    # Rebuild top
    x = layers.GlobalAveragePooling2D(name="avg_pool")(model.output)
    x = layers.BatchNormalization()(x)

    top_dropout_rate = 0.2
    x = layers.Dropout(top_dropout_rate, name="top_dropout")(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax", name="pred")(x)

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        0.0001,
        decay_steps=100,
        decay_rate=0.5,
        staircase=False
    )

    # Compile
    model = tf.keras.Model(inputs, outputs, name="EfficientNet")
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])
    return model


with strategy.scope():
    model = build_model(num_classes=NUM_CLASSES)

epochs = 1000  # @param {type: "slider", min:8, max:80}
hist = model.fit(train_ds, epochs=epochs, validation_data=val_ds, verbose=0, callbacks=[tensorboard_callback, cp_callback])
print("Average loss: ", np.average(hist.history['loss']))


# # unfreeze layers again
# def unfreeze_model(model):
#     # We unfreeze the top 20 layers while leaving BatchNorm layers frozen
#     for layer in model.layers[-20:]:
#         if not isinstance(layer, layers.BatchNormalization):
#             layer.trainable = True
#
#     optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
#     model.compile(
#         optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"]
#     )
#
#
# unfreeze_model(model)
#
# epochs = 10  # @param {type: "slider", min:8, max:50}
# hist = model.fit(train_ds, epochs=epochs, validation_data=val_ds, verbose=0, callbacks=[tensorboard_callback, cp_callback])
# print("Average loss: ", np.average(hist.history['loss']))
