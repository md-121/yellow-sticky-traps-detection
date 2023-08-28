# splits images and csv-based annotations into patches

import csv
import os
import timeit

from matplotlib import image

image_size_x = 5184
image_size_y = 3456

patch_size_x = 800
patch_size_y = 1320

overlap_size = 252

IMAGE_PATH = "./Images/val/"
ANN_PATH = "./Annotations/val_label.csv"
SAVE_IMG_DIR = "./Patched_Images/val/"
SAVE_ANN_DIR = "./Patched_Annotations/val_label.csv"


def calc_start_points():
    x = patch_size_x
    patch_sections_x = []
    while x <= image_size_x:
        patch_sections_x.append(x)
        x = x - overlap_size + patch_size_x

    y = patch_size_y
    patch_sections_y = []
    while y <= image_size_y:
        patch_sections_y.append(y)
        y = y - overlap_size + patch_size_y

    x_patches = []
    for patch in patch_sections_x:
        x_patches.append((patch - patch_size_x, patch))

    y_patches = []
    for patch in patch_sections_y:
        y_patches.append((patch - patch_size_y, patch))

    return x_patches, y_patches


def patch_image(x_patches, y_patches):
    for file in os.listdir(IMAGE_PATH):
        if file.endswith(".jpg"):
            print(file)
            img = image.imread(IMAGE_PATH + file)
            patch_count = 0
            for x_patch in x_patches:
                for y_patch in y_patches:
                    patch = img[y_patch[0]:y_patch[1], x_patch[0]:x_patch[1], :]

                    f_name = file.split(".")[0]
                    image.imsave(SAVE_IMG_DIR + f_name + "_" + str(patch_count) + ".png", patch)

                    patch_count = patch_count + 1


def patch_annotations(x_patches, y_patches):
    csv_rows = []
    header = None
    with open(ANN_PATH, 'r') as csv_file:
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

        adapted_rows = [header]

        patch_count = 0
        for x_patch in x_patches:
            for y_patch in y_patches:
                for r in csv_rows:
                    if x_patch[0] <= int(r[4]) < x_patch[1] and x_patch[0] <= int(r[6]) < x_patch[1] and y_patch[0] <= int(r[5]) < y_patch[1] and y_patch[0] <= int(r[7]) < y_patch[1]:
                        r_tmp = r.copy()
                        f_name = r_tmp[0].split(".")[0]
                        r_tmp[0] = f_name + "_" + str(patch_count) + ".png"
                        r_tmp[1] = str(patch_size_x)
                        r_tmp[2] = str(patch_size_y)
                        r_tmp[4] = str(int(r_tmp[4]) - x_patch[0])
                        r_tmp[5] = str(int(r_tmp[5]) - y_patch[0])
                        r_tmp[6] = str(int(r_tmp[6]) - x_patch[0])
                        r_tmp[7] = str(int(r_tmp[7]) - y_patch[0])
                        adapted_rows.append(r_tmp)
                patch_count = patch_count + 1

        with open(SAVE_ANN_DIR, 'a') as file:
            writer = csv.writer(file)
            writer.writerows(adapted_rows)


if __name__ == "__main__":
    start = timeit.default_timer()

    x_patches, y_patches = calc_start_points()
    patch_image(x_patches, y_patches)
    patch_annotations(x_patches, y_patches)

    stop = timeit.default_timer()
    print('Runtime: ', stop - start)
