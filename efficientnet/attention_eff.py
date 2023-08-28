import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras import backend as K
from tf_keras_vis.saliency import Saliency
from tf_keras_vis.utils import normalize
from tf_keras_vis.gradcam import GradcamPlusPlus


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

# Image titles
image_titles = ['wf', 'mr', 'nc']

# Load images
img1 = load_img('images/load/wf.png', target_size=IMG_SIZE)
img2 = load_img('images/load/mr.png', target_size=IMG_SIZE)
img3 = load_img('images/load/nc.png', target_size=IMG_SIZE)
images = np.asarray([np.array(img1), np.array(img2), np.array(img3)])

# Preparing input data
X = preprocess_input(images)

# Rendering
subplot_args = { 'nrows': 1, 'ncols': 3, 'figsize': (9, 3), 'subplot_kw': {'xticks': [], 'yticks': []} }
f, ax = plt.subplots(**subplot_args)
for i, title in enumerate(image_titles):
    ax[i].set_title(title, fontsize=14)
    ax[i].imshow(images[i])
plt.tight_layout()
plt.savefig('images/input.png')


# The `output` variable refer to the output of the model,
# so, in this case, `output` shape is `(3, 1000)` i.e., (samples, classes).
def loss(output):
    # 1 is the imagenet index corresponding to Goldfish, 294 to Bear and 413 to Assault Rifle.
    return (output[0][0], output[1][1], output[2][2])


def model_modifier(m):
    m.layers[-1].activation = tf.keras.activations.linear
    return m

from tf_keras_vis.scorecam import ScoreCAM

# Create ScoreCAM object
scorecam = ScoreCAM(model, model_modifier, clone=False)

cam = scorecam(loss,
               X,
               penultimate_layer=-1, # model.layers number
              )
cam = normalize(cam)

f, ax = plt.subplots(**subplot_args)
for i, title in enumerate(image_titles):
    heatmap = np.uint8(cm.jet(cam[i])[..., :3] * 255)
    ax[i].set_title(title, fontsize=14)
    ax[i].imshow(images[i])
    ax[i].imshow(heatmap, cmap='jet', alpha=0.5)
plt.tight_layout()
plt.savefig('images/sc.png')


gradcam = GradcamPlusPlus(model, model_modifier, clone=False)

# Generate heatmap with GradCAM++
cam = gradcam(loss, X, penultimate_layer=-1)
cam = normalize(cam)

f, ax = plt.subplots(**subplot_args)
for i, title in enumerate(image_titles):
    heatmap = np.uint8(cm.get_cmap('jet')(cam[i])[..., :3] * 255)
    ax[i].set_title(title, fontsize=14)
    ax[i].imshow(images[i])
    ax[i].imshow(heatmap, cmap='jet', alpha=0.8)
plt.tight_layout()
plt.savefig('images/gradcam_plus_plus.png')
