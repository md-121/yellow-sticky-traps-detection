import csv

import cv2
from bounding_box import bounding_box as bb

from misc.annotation_code import AnnotationCode
from misc.defaults import DATASET_PATH, ANNOTATIONS_SUBDIR, IMAGES_FILE_EXT, IMAGES_SUBDIR, \
    ANNOTATIONS_FILE

# Local Settings
DISPLAY_IMGS = True
DISP_IMG_NAME = "1001_3"
SKIP_TR = False
DEBUG = False


def show_annotations(image_path, annotations):
    print(annotations)
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)

    for annotation in annotations:
        label = AnnotationCode(annotation[0]).name
        bb.add(image, annotation[1], annotation[2], annotation[3], annotation[4], label)

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

    header = []
    csv_rows = []
    with open(DATASET_PATH + ANNOTATIONS_SUBDIR + ANNOTATIONS_FILE, 'r') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                line_count += 1
                header = row
            else:
                csv_rows.append(row)
                line_count += 1
        print(f'Processed {line_count} lines.')

        annotation_dict = {}

        if DEBUG:
            print(DATASET_PATH + ANNOTATIONS_SUBDIR + ANNOTATIONS_FILE)

        for row in csv_rows:
            obj_list = []
            img_name = row[0].split('.')[0]

            if img_name not in annotation_dict:
                annotation_dict[img_name] = []

            # class
            name = row[3]
            if name == "TR":
                if not SKIP_TR:
                    obj_list.append(AnnotationCode[name].value)

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

            # bounding box
            for i in range(4, 8):
                obj_list.append(int(row[i]))

            annotation_dict[img_name].append(obj_list)

        # display result
        for img_name, annotation_list in annotation_dict.items():
            if DEBUG:
                print(annotation_list)
            if DISPLAY_IMGS:
                if DISP_IMG_NAME is not None:
                    if img_name != DISP_IMG_NAME:
                        continue
                show_annotations(DATASET_PATH + IMAGES_SUBDIR + img_name + IMAGES_FILE_EXT, annotation_list)

    # statistics
    print("Statistics:" + "\n" + "WF: " + str(wf_count) + "\n" + "MR: " + str(mr_count) + "\n" + "NC: " + str(
        nc_count) + "\n" + "TR: " + str(
        tr_count) + "\n\n" + "Total: " + str(total_count) + "\n" + "Without TR: " + str(
        wo_tr_count) + "\n" + "Error: " + str(error_count) + "\n")
