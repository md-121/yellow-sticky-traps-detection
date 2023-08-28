# https://github.com/Devyanshu/image-split-with-overlap/blob/master/split_image_with_overlap.py
# NOT USED
import cv2

path_to_img = ""
split_width = 150
split_height = 150

img = cv2.imread(path_to_img)
img_h, img_w, _ = img.shape

def start_points(size, split_size, overlap=0):
    points = [0]
    stride = int(split_size * (1-overlap))
    counter = 1
    while True:
        pt = stride * counter
        if pt + split_size >= size:
            print("No!")
            points.append(size - split_size)
            break
        else:
            points.append(pt)
        counter += 1
    return points


X_points = start_points(img_w, split_width, 0.5)
Y_points = start_points(img_h, split_height, 0.5)

# count = 0
# name = 'splitted'
# frmt = 'jpeg'
#
# for i in Y_points:
#     for j in X_points:
#         split = img[i:i+split_height, j:j+split_width]
#         cv2.imwrite('{}_{}.{}'.format(name, count, frmt), split)
#         count += 1