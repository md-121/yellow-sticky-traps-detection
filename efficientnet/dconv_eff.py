# https://keras.io/examples/vision/image_classification_efficientnet_fine_tuning/

import tensorflow as tf
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

test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    DATASET_PATH + "test/",
    label_mode="categorical",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

test_ds = test_ds.prefetch(tf.data.experimental.AUTOTUNE)


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


def compute_loss(input_image, filter_index):
    activation = feature_extractor(input_image)
    # We avoid border artifacts by only involving non-border pixels in the loss.
    filter_activation = activation[:, 2:-2, 2:-2, filter_index]
    return tf.reduce_mean(filter_activation)


@tf.function
def gradient_ascent_step(img, filter_index, learning_rate):
    with tf.GradientTape() as tape:
        tape.watch(img)
        loss = compute_loss(img, filter_index)
    # Compute gradients.
    grads = tape.gradient(loss, img)
    # Normalize gradients.
    grads = tf.math.l2_normalize(grads)
    img += learning_rate * grads
    return loss, img


def initialize_image():
    # We start from a gray image with some random noise
    img = tf.random.uniform((1, IMG_SIZE[0], IMG_SIZE[1], 3))
    # ResNet50V2 expects inputs in the range [-1, +1].
    # Here we scale our random inputs to [-0.125, +0.125]
    return img * 0.25


def visualize_filter(filter_index):
    # We run gradient ascent for 20 steps
    iterations = 30
    learning_rate = 10.0
    img = initialize_image()
    for iteration in range(iterations):
        loss, img = gradient_ascent_step(img, filter_index, learning_rate)

    # Decode the resulting input image
    img = deprocess_image(img[0].numpy())
    return loss, img


def deprocess_image(img):
    # Normalize array: center on 0., ensure variance is 0.15
    img -= img.mean()
    img /= img.std() + 1e-5
    img *= 0.15

    # Center crop
    img = img[25:-25, 25:-25, :]

    # Clip to [0, 1]
    img += 0.5
    img = np.clip(img, 0, 1)

    # Convert to RGB array
    img *= 255
    img = np.clip(img, 0, 255).astype("uint8")
    return img



with strategy.scope():
    model = build_model(num_classes=NUM_CLASSES)

layer_name = "block3b_project_conv"

# Set up a model that returns the activation values for our target layer
layer = model.get_layer(name=layer_name)
feature_extractor = tf.keras.Model(inputs=model.inputs, outputs=layer.output)

loss, img = visualize_filter(10)
tf.keras.preprocessing.image.save_img("0.png", img)
