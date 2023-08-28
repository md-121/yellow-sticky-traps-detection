# Code to cut the GT Bounding Box out of the image to train classifier

import os
import xml.etree.ElementTree as et

import cv2
from bounding_box import bounding_box as bb

from misc.annotation_code import AnnotationCode
from misc.defaults import DATASET_PATH, ANNOTATIONS_SUBDIR, ANNOTATIONS_FILE_EXT, IMAGES_FILE_EXT, IMAGES_SUBDIR


def show_annotations(image_path, annotations):
    img_name = image_path.split("/")[-1]
    img_name = img_name.split('.')[0]
    print(img_name)
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)

    counter = 0
    for annotation in annotations:
        counter = counter + 1
        print(annotation)
        label = AnnotationCode(annotation[0]).name
        crop_img = image[annotation[3]:annotation[5], annotation[2]:annotation[4]]
        cv2.imwrite("./Eff_Dataset/GT_cut/" + label + "/" + img_name + "_" + str(counter) + ".png", crop_img)
        counter = counter + 1


if __name__ == "__main__":
    wf_count = 0
    mr_count = 0
    nc_count = 0
    tr_count = 0

    total_count = 0
    wo_tr_count = 0

    error_count = 0

    for file in os.listdir(DATASET_PATH + ANNOTATIONS_SUBDIR):
        if file.endswith(ANNOTATIONS_FILE_EXT):
            annotation_list = []  # [class, truncated, xmin, ymin, xmax, ymax]
            tr_found = False

            root = et.parse(DATASET_PATH + ANNOTATIONS_SUBDIR + file).getroot()
            for obj in root.iter('object'):
                obj_list = []

                # class
                name = obj.find('name').text
                if name == "TR":
                    obj_list.append(AnnotationCode[name].value)

                    tr_found = True
                    tr_count = tr_count + 1
                    total_count = total_count + 1
                elif name == "WF":
                    obj_list.append(AnnotationCode[name].value)

                    wf_count = wf_count + 1
                    total_count = total_count + 1
                    wo_tr_count = wo_tr_count + 1
                elif name == "MR":
                    obj_list.append(AnnotationCode[name].value)

                    mr_count = mr_count + 1
                    total_count = total_count + 1
                    wo_tr_count = wo_tr_count + 1
                elif name == "NC":
                    obj_list.append(AnnotationCode[name].value)

                    nc_count = nc_count + 1
                    total_count = total_count + 1
                    wo_tr_count = wo_tr_count + 1
                else:
                    obj_list.append(AnnotationCode[name].value)

                    error_count = error_count + 1

                # is obj truncated
                truncated = obj.find('truncated').text
                obj_list.append(int(truncated))

                # bounding box
                bndbox = obj.find('bndbox')
                for child in bndbox:
                    obj_list.append(int(child.text))

                annotation_list.append(obj_list)

            img_name = file.split('.')[0]
            show_annotations(DATASET_PATH + IMAGES_SUBDIR + img_name + IMAGES_FILE_EXT, annotation_list)

    # statistics
    print("Statistics:" + "\n" + "WF: " + str(wf_count) + "\n" + "MR: " + str(mr_count) + "\n" + "NC: " + str(
        nc_count) + "\n" + "TR: " + str(
        tr_count) + "\n\n" + "Total: " + str(total_count) + "\n" + "Without TR: " + str(
        wo_tr_count) + "\n" + "Error: " + str(error_count) + "\n")
