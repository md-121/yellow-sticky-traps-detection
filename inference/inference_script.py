#!/usr/bin/env python
# coding: utf-8
# https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/auto_examples/plot_object_detection_checkpoint.html#sphx-glr-auto-examples-plot-object-detection-checkpoint-py
# Object Detection From TF2 Checkpoint
import csv
import os
import time
import warnings
from enum import Enum

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import scipy.optimize as sp
from PIL import Image
from object_detection.builders import model_builder
from object_detection.utils import config_util
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils


# SWITCHES
VIS_BBOXES = True  # Visualize Bounding Boxes in Images
SCORE_THRES = 0.9


class AnnotationCode(Enum):
    WF = 0  # Whitefly
    MR = 1  # Macrolophus
    NC = 2  # Nesidiocoris


# https://gist.github.com/AruniRC/c629c2df0e68e23aff7dcaeef87c72d4#file-bbox_iou_evaluation-py-L32
def bbox_iou(boxA, boxB):
    # https://www.pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/
    # ^^ corrected.

    # Determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interW = xB - xA + 1
    interH = yB - yA + 1

    # Correction: reject non-overlapping boxes
    if interW <= 0 or interH <= 0:
        return -1.0

    interArea = interW * interH
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou


def match_bboxes(bbox_gt, bbox_pred, IOU_THRESH=0.5):
    '''
    Given sets of true and predicted bounding-boxes,
    determine the best possible match.
    Parameters
    ----------
    bbox_gt, bbox_pred : N1x4 and N2x4 np array of bboxes [x1,y1,x2,y2].
      The number of bboxes, N1 and N2, need not be the same.

    Returns
    -------
    (idxs_true, idxs_pred, ious, labels)
        idxs_true, idxs_pred : indices into gt and pred for matches
        ious : corresponding IOU value of each match
        labels: vector of 0/1 values for the list of detections
    '''
    n_true = bbox_gt.shape[0]
    n_pred = bbox_pred.shape[0]
    MAX_DIST = 1.0
    MIN_IOU = 0.0

    # NUM_GT x NUM_PRED
    iou_matrix = np.zeros((n_true, n_pred))
    for i in range(n_true):
        for j in range(n_pred):
            iou_matrix[i, j] = bbox_iou(bbox_gt[i, :], bbox_pred[j, :])

    if n_pred > n_true:
        # there are more predictions than ground-truth - add dummy rows
        diff = n_pred - n_true
        iou_matrix = np.concatenate((iou_matrix,
                                     np.full((diff, n_pred), MIN_IOU)),
                                    axis=0)

    if n_true > n_pred:
        # more ground-truth than predictions - add dummy columns
        diff = n_true - n_pred
        iou_matrix = np.concatenate((iou_matrix,
                                     np.full((n_true, diff), MIN_IOU)),
                                    axis=1)

    # call the Hungarian matching
    idxs_true, idxs_pred = sp.linear_sum_assignment(1 - iou_matrix)

    if (not idxs_true.size) or (not idxs_pred.size):
        ious = np.array([])
    else:
        ious = iou_matrix[idxs_true, idxs_pred]

    # remove dummy assignments
    sel_pred = idxs_pred < n_pred
    idx_pred_actual = idxs_pred[sel_pred]
    idx_gt_actual = idxs_true[sel_pred]
    ious_actual = iou_matrix[idx_gt_actual, idx_pred_actual]
    sel_valid = (ious_actual > IOU_THRESH)
    label = sel_valid.astype(int)

    return idx_gt_actual[sel_valid], idx_pred_actual[sel_valid], ious_actual[sel_valid], label


def load_gt():
    csv_rows = []
    with open("./Annotations/test_label.csv", 'r') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                line_count += 1
            else:
                csv_rows.append(row)
                line_count += 1
        print(f'Processed {line_count} lines.')

        annotation_dict = {}
        label_dict = {}

        for row in csv_rows:
            obj_list = []
            img_name = row[0].split('.')[0]

            if img_name not in annotation_dict:
                annotation_dict[img_name] = []

            if img_name not in label_dict:
                label_dict[img_name] = []

            # bounding box
            for b in range(4, 8):
                obj_list.append(int(row[b]))

            annotation_dict[img_name].append(obj_list)
            label_dict[img_name].append(row[3])
    return annotation_dict, label_dict


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow logging (1)
tf.get_logger().setLevel('ERROR')  # Suppress TensorFlow logging (2)

# Enable GPU dynamic memory allocation
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


def load_images():
    base_path = './datasets/Dataset_Patched/Images/test/'
    filenames = []
    for file in os.listdir(base_path):
        if file.endswith(".png"):
            filenames.append(base_path + file)

    return filenames


IMAGE_PATHS = load_images()

PATH_TO_MODEL_DIR = './faster_rcnn_inception_resnet_v2_1024x1024_coco17_tpu-8/'
PATH_TO_LABELS = "./Annotations/label_map.pbtxt"

# Load the model
PATH_TO_CFG = PATH_TO_MODEL_DIR + "pipeline.config"
PATH_TO_CKPT = PATH_TO_MODEL_DIR + "checkpoints/"

print('Loading model... ', end='')
start_time = time.time()

# Load pipeline config and build a detection model
configs = config_util.get_configs_from_pipeline_file(PATH_TO_CFG)
model_config = configs['model']
detection_model = model_builder.build(model_config=model_config, is_training=False)

# Restore checkpoint
ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore(os.path.join(PATH_TO_CKPT, 'ckpt-201')).expect_partial()


@tf.function
def detect_fn(image):
    """Detect objects in image."""

    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)

    return detections


end_time = time.time()
elapsed_time = end_time - start_time
print('Done! Took {} seconds'.format(elapsed_time))

# Load label map data (for plotting)
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS,
                                                                    use_display_name=True)

warnings.filterwarnings('ignore')  # Suppress Matplotlib warnings


def load_image_into_numpy_array(path):
    """Load an image from file into a numpy array.

    Puts image into numpy array to feed into tensorflow graph.
    Note that by convention we put it into a numpy array with shape
    (height, width, channels), where channels=3 for RGB.

    Args:
      path: the file path to the image

    Returns:
      uint8 numpy array with shape (img_height, img_width, 3)
    """
    return np.array(Image.open(path))[:, :, :3]


bboxes_gt, labels_gt = load_gt()
matched_boxes = {}

mr_pred, nc_pred, wf_pred = 0, 0, 0

for image_path in IMAGE_PATHS:
    img_name = image_path.split("/")[-1]
    img_name_wosub = img_name.split(".")[0]
    print('Running inference for {}... '.format(image_path), end='')

    image_np = load_image_into_numpy_array(image_path)

    # Things to try:
    # Flip horizontally
    # image_np = np.fliplr(image_np).copy()

    # Convert image to grayscale
    # image_np = np.tile(
    #     np.mean(image_np, 2, keepdims=True), (1, 1, 3)).astype(np.uint8)

    input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)

    detections = detect_fn(input_tensor)

    # All outputs are batches tensors.
    # Convert to numpy arrays, and take index [0] to remove the batch dimension.
    # We're only interested in the first num_detections.
    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                  for key, value in detections.items()}
    detections['num_detections'] = num_detections

    # detection_classes should be ints.
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

    label_id_offset = 1

    if VIS_BBOXES:
        image_np_with_detections = image_np.copy()

        viz_utils.visualize_boxes_and_labels_on_image_array(
            image_np_with_detections,
            detections['detection_boxes'],
            detections['detection_classes'] + label_id_offset,
            detections['detection_scores'],
            category_index,
            use_normalized_coordinates=True,
            max_boxes_to_draw=200,
            min_score_thresh=.901,
            agnostic_mode=False)

        plt.imsave(img_name, image_np_with_detections)

    # calc IoU
    bbox_pred = []
    label_pred = []
    score_pred = []
    for i in range(len(detections['detection_boxes'])):
        score = detections['detection_scores'][i]
        if score <= SCORE_THRES:
            continue

        img_shape = image_np.shape
        box = detections['detection_boxes'][i]
        ymin = box[0] * img_shape[0]
        xmin = box[1] * img_shape[1]
        ymax = box[2] * img_shape[0]
        xmax = box[3] * img_shape[1]

        bbox_pred.append([xmin, ymin, xmax, ymax])
        label_pred.append(detections['detection_classes'][i])
        score_pred.append(score)

    for l_pred in label_pred:
        if l_pred == 0:
            wf_pred = wf_pred + 1
        elif l_pred == 1:
            mr_pred = mr_pred + 1
        elif l_pred == 2:
            nc_pred = nc_pred + 1

    bbox_gt = []
    label_gt = []
    if img_name_wosub in bboxes_gt:
        bbox_gt = bboxes_gt[img_name_wosub]
        label_gt = labels_gt[img_name_wosub]

    idx_gt_actual, idx_pred_actual, ious_actual, label = match_bboxes(np.array(bbox_gt), np.array(bbox_pred))

    # find IoU valid elements and collect them
    matches = []
    for i in range(len(idx_pred_actual)):
        gt = idx_gt_actual[i]
        pred = idx_pred_actual[i]

        gt_label = AnnotationCode[label_gt[gt]].value
        pred_label = label_pred[pred]
        pred_score = score_pred[pred]

        matches.append([gt_label, pred_label, pred_score])
    print('Done.')
    matched_boxes[img_name_wosub] = matches

# calc eval
# building matrix
# i = gt, j = pred
# wf, mr, nc
eval_mat = np.zeros((3, 3))

for _, matches in matched_boxes.items():
    for match in matches:
        if match[2] > SCORE_THRES:
            gt = match[0]
            pred = match[1]
            print("GT: " + str(gt))
            print("Pred: " + str(pred))
            eval_mat[gt][pred] = eval_mat[gt][pred] + 1

total_tp = eval_mat[0][0] + eval_mat[1][1] + eval_mat[2][2]

mr_cnt = 0
nc_cnt = 0
wf_cnt = 0
for _, img in labels_gt.items():
    for code in img:
        val = AnnotationCode[code].value
        if val == 0:
            wf_cnt = wf_cnt + 1
        elif val == 1:
            mr_cnt = mr_cnt + 1
        elif val == 2:
            nc_cnt = nc_cnt + 1

w_class_acc = total_tp / (mr_cnt + nc_cnt + wf_cnt) * 100

print("Total GTs:")
print("MR GTs: " + str(mr_cnt))
print("NC GTs: " + str(nc_cnt))
print("WF GTs: " + str(wf_cnt))
print("Total Preds:")
print("MR Pred: " + str(mr_pred))
print("NC Pred: " + str(nc_pred))
print("WF Pred: " + str(wf_pred))
print("Weighted Classification Accuracy: " + str(w_class_acc))
print("\nEval Matrix: " + str(eval_mat) + "\n")

print("Processing finished!")
