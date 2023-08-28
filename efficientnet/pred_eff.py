# https://keras.io/examples/vision/image_classification_efficientnet_fine_tuning/

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.applications import EfficientNetB3

# IMG_SIZE is determined by EfficientNet model choice
IMG_SIZE = (300, 300)
BATCH_SIZE = 1
BUFFER_SIZE = 32
NUM_CLASSES = 3
DATASET_PATH = "./datasets/Real_cut/"
CKPT_PATH = "./checkpoints/cp.ckpt"
TRAIN_PATH = "./training/efficientnet-b3_1/"

strategy = tf.distribute.MirroredStrategy()

test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    DATASET_PATH,
    label_mode=None,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# Transfer learning
def build_model(num_classes):
    inputs = layers.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
    model = EfficientNetB3(include_top=False, input_tensor=inputs, weights=None)

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

res = model.predict(x=test_ds, verbose=0)
print(str(res))
