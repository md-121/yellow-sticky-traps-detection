from matplotlib import pyplot as plt
import tensorflow as tf
from tf_keras_vis.activation_maximization import ActivationMaximization
from tf_keras_vis.utils.callbacks import Print
from tensorflow.keras import layers
from tensorflow.keras.applications import EfficientNetB0
import numpy as np


# IMG_SIZE is determined by EfficientNet model choice
IMG_SIZE = (224, 224)
BATCH_SIZE = 64
BUFFER_SIZE = 32
NUM_CLASSES = 3
DATASET_PATH = "./datasets/GT_cut/"
CKPT_PATH = "./checkpoints/cp.ckpt"
TRAIN_PATH = "./training/efficientnet-b0_3/"

strategy = tf.distribute.MirroredStrategy()


# Transfer learning
def build_model(num_classes):
    inputs = layers.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
    model = EfficientNetB0(include_top=False, input_tensor=inputs, weights=None)

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
    model.load_weights(CKPT_PATH)
    return model


with strategy.scope():
    model = build_model(num_classes=NUM_CLASSES)

layer_name = 'top_conv' # The target layer that is the last layer of VGG16.


def model_modifier(current_model):
    target_layer = current_model.get_layer(name=layer_name)
    new_model = tf.keras.Model(inputs=current_model.inputs, outputs=target_layer.output)
    new_model.layers[-1].activation = tf.keras.activations.linear
    return new_model


def loss(output):
    return output[..., filter_number]


for i in range(0, 1280):
    filter_number = i
    activation_maximization = ActivationMaximization(model, model_modifier, clone=False)

    # Generate max activation
    activation = activation_maximization(loss, callbacks=[Print(interval=50)])
    image = activation[0].astype(np.uint8)

    tf.keras.preprocessing.image.save_img("vis/" + str(filter_number) + ".png", image)
