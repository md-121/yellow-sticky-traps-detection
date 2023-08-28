import os
import xml.etree.ElementTree as et

import cv2
from bounding_box import bounding_box as bb

from misc.annotation_code import AnnotationCode
from misc.defaults import DATASET_PATH, ANNOTATIONS_SUBDIR, ANNOTATIONS_FILE_EXT, IMAGES_FILE_EXT, IMAGES_SUBDIR

# Local Settings
DISPLAY_IMGS = True
DISPLAY_ONLY_TR = False
SKIP_TR = False
DEBUG = False


def show_annotations(image_path, annotations):
    if image_path.split("/")[-1] != "1271.jpg":
        return
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)

    for annotation in annotations:
        print(annotation)
        label = AnnotationCode(annotation[0]).name
        bb.add(image, annotation[2], annotation[3], annotation[4], annotation[5], label)

    cv2.imshow(image_path, image)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()


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

            if DEBUG:
                print(DATASET_PATH + ANNOTATIONS_SUBDIR + file)

            root = et.parse(DATASET_PATH + ANNOTATIONS_SUBDIR + file).getroot()
            for obj in root.iter('object'):
                obj_list = []

                # class
                name = obj.find('name').text
                if name == "TR":
                    if not SKIP_TR:
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

            # display result
            if DEBUG:
                print(annotation_list)
            if DISPLAY_IMGS:
                if not DISPLAY_ONLY_TR:
                    tr_found = True
                if tr_found:
                    img_name = file.split('.')[0]
                    show_annotations(DATASET_PATH + IMAGES_SUBDIR + img_name + IMAGES_FILE_EXT, annotation_list)

    # statistics
    print("Statistics:" + "\n" + "WF: " + str(wf_count) + "\n" + "MR: " + str(mr_count) + "\n" + "NC: " + str(
        nc_count) + "\n" + "TR: " + str(
        tr_count) + "\n\n" + "Total: " + str(total_count) + "\n" + "Without TR: " + str(
        wo_tr_count) + "\n" + "Error: " + str(error_count) + "\n")
