# https://www.tensorflow.org/tutorials/load_data/images#load_using_tfdata
# https://www.tensorflow.org/tutorials/images/transfer_learning#top_of_page
# https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md
# https://github.com/tensorflow/models/tree/master/research/object_detection
# https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/training.html

import tensorflow as tf

from misc.defaults import IMAGES_SUBDIR, DATASET_PATH

AUTOTUNE = tf.data.experimental.AUTOTUNE

import os
import pathlib

BATCH_SIZE = 32
IMG_HEIGHT = 224
IMG_WIDTH = 224

ANNOTATIONS = ["WF", "MR", "NC"]


def get_label(file_path):
    # convert the path to a list of path components
    parts = tf.strings.split(file_path, os.path.sep)
    # The second to last is the class-directory
    return parts[-2] == CLASS_NAMES


def decode_img(img):
    # convert the compressed string to a 3D uint8 tensor
    img = tf.image.decode_jpeg(img, channels=3)
    # Use `convert_image_dtype` to convert to floats in the [0,1] range.
    img = tf.image.convert_image_dtype(img, tf.float32)
    # resize the image to the desired size.
    return tf.image.resize(img, [IMG_HEIGHT, IMG_WIDTH])


def process_path(file_path):
    print()
    label = get_label(file_path)
    # load the raw data from the file as a string
    img = tf.io.read_file(file_path)
    img = decode_img(img)
    return img, label


if __name__ == "__main__":
    tf.enable_eager_execution()
    #tf.compat.v1.enable_eager_execution()

    #print(tf.__version__)

    data_dir = pathlib.Path(DATASET_PATH + IMAGES_SUBDIR)
    list_ds = tf.data.Dataset.list_files(str(data_dir / '*.jpg'))  # TODO: remove hardcoded file type

    # for f in list_ds.take(5):
    #     print(f.numpy())

    labeled_ds = list_ds.map(process_path, num_parallel_calls=AUTOTUNE)

