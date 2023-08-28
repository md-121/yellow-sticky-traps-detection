# https://datacarpentry.org/image-processing/07-thresholding/
"""
 * Python script to demonstrate adaptive thresholding using Otsu's method.
 *
 * usage: python AdaptiveThreshold.py <filename> <sigma>
"""
import os
from collections import deque

import numpy as np
import skimage.filters
from PIL import Image


def find_fg_neighbors(a: np.ndarray, pos: tuple) -> list:
    """
    Searches foreground neighbors of a given position.

    :param a: Matrix to search in.
    :param pos: Current position for neighbor search.
    :return: List of positions of foreground neighbors.
    """

    fg_neighbors = []
    i_border, j_border = a.shape
    i_pos, j_pos = pos

    # get all 8-connected neighbors.
    # max and min functions are used to stay inside matrix bounds.
    for i in range(max(0, i_pos - 1), min(i_border, i_pos + 2)):
        for j in range(max(0, j_pos - 1), min(j_border, j_pos + 2)):
            if (i_pos, j_pos) == (i, j):  # ignore current position
                continue
            if a[i, j] == 1:
                fg_neighbors.append((i, j))  # add foreground neighbor coordinates.

    return fg_neighbors


def next_pos(a: np.ndarray, pos: tuple) -> tuple:
    """
    Calculates the next position as seen from a given position.

    :param a: Matrix to traverse.
    :param pos: Current position from where the next position is searched.
    :return: Next position in the matrix.
    """

    i_border, j_border = a.shape
    i_pos, j_pos = pos

    # check to maintain matrix bounds.
    if i_pos + 1 >= i_border:
        if j_pos + 1 >= j_border:
            return -1, -1  # last position reached. Returns abort condition.
        else:
            return 0, j_pos + 1  # i_pos reached bound. Reset it to 0 and increase j_pos.
    else:
        return i_pos + 1, j_pos  # increase i_pos till reaching bound.


def connected_components(a: np.ndarray) -> np.ndarray:
    """
    Finds connected-components and labels them.
    Uses the 'One component at a time' approach.

    :param a: Matrix to search connected components in.
    :return: Matrix with shape of a containing labeled areas.
    """

    current_pos = (0, 0)  # init start position
    current_label = 1  # start label
    label_array = np.zeros_like(a)

    stack = deque()  # stack containing positions for foreground neighborhood (connected-components) check

    while current_pos != (-1, -1):  # (-1, -1) is the abort condition when reaching the end of the matrix
        elem = a[current_pos]

        if elem > 0 and label_array[current_pos] == 0:  # check if the element is in foreground and not labeled yet
            label_array[current_pos] = current_label  # label the current element
            stack.append(current_pos)  # push to stack to search its neighborhood for foreground (connected) elements.

            while len(stack) != 0:  # run for all neighbours (until stack is empty)
                pos = stack.pop()
                fns = find_fg_neighbors(a, pos)  # find all foreground neighbors

                # set INCLUDE_SINGLE_ELEMENT_REGIONS to False if regions of size 1
                # should be excluded and labeled as background
                if not INCLUDE_SINGLE_ELEMENT_REGIONS:
                    if len(fns) == 0:
                        label_array[pos] = 0  # has no neighbor so not labeling it

                # label all foreground neighbors which are not labeled yet
                for fn in fns:
                    if label_array[fn] == 0:
                        label_array[fn] = current_label
                        stack.append(fn)

            # area completely labeled.
            # move to next position and increase label.
            current_label = current_label + 1
            current_pos = next_pos(a, current_pos)
        else:
            # element is background or already labeled
            # go to next position
            current_pos = next_pos(a, current_pos)

    return label_array


if __name__ == "__main__":
    for filename in os.listdir("./Segment_Test_Set/Full/"):
        if filename.endswith(".jpg"):
            filename_wo_suf = filename.split(".")[0]
            sigma = float(2.0)

            # read and display the original image
            image = Image.open("./Segment_Test_Set/Full/" + filename)
            sat_image = image.convert(mode="HSV")

            image = np.array(image)

            sat_image = np.array(sat_image)
            sat_image = sat_image[:, :, 1]

            # blur and grayscale before thresholding
            blur = skimage.filters.gaussian(sat_image, sigma=sigma)

            # perform adaptive thresholding
            mask = blur <= (1.0 / 255.0) * 241.0

            # use the mask to select the "interesting" part of the image
            sel = np.zeros_like(image)
            sel[mask] = image[mask]
            INCLUDE_SINGLE_ELEMENT_REGIONS = False

            # generate labeled areas
            label_arr = connected_components(mask.astype(int))

            for i in range(1, np.max(label_arr) + 1):
                ind = np.argwhere(label_arr == i)
                if len(ind) == 0:
                    continue
                area_max = np.amax(ind, axis=0)
                area_min = np.amin(ind, axis=0)

                x_size = area_max[1] + 1 - area_min[1]
                y_size = area_max[0] + 1 - area_min[0]

                # throw out too small images
                area_size = x_size * y_size
                if area_size <= 435:
                    print("Skipping too small image (size, ID): (" + str(area_size) + ", " + str(i) + ")")
                    continue
                elif area_size >= 40000:
                    print("Skipping too large image (size, ID): (" + str(area_size) + ", " + str(i) + ")")
                    continue

                cut_out = image[area_min[0]:area_max[0] + 1, area_min[1]:area_max[1] + 1]
                Image.fromarray(cut_out).save("./Segment_Test_Set/Cut/" + filename_wo_suf + "_" + str(i) + ".png")

    print("Done.")
