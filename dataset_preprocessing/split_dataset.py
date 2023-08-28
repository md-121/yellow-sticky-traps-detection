# splits the images and csv-based annotations into train/test/val

import csv
import os
import random
from shutil import copyfile

NUM_IMGS = 284
NUM_TRAIN_IMGS = 199
NUM_TEST_IMGS = 57
NUM_VAL_IMGS = 28

IMGS_PATH = "./YellowStickTraps_Dataset/Images/"
IMG_SAVE_PATH = "./Eff_Dataset/Images/"
IMG_FILE_ENDING = ".jpg"

ANN_PATH = "./YellowStickTraps_Dataset/labels.csv"
ANN_SAVE_PATH = "./Eff_Dataset/Annotations/"

imgs = []
for file in os.listdir(IMGS_PATH):
    if file.endswith(IMG_FILE_ENDING):
        imgs.append(file)

header = []
csv_rows = []
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

print(f'Found {len(imgs)} images.\nSplitting up the dataset...')

train_imgs = []
test_imgs = []
val_imgs = []

train_ann = [header]
test_ann = [header]
val_ann = [header]
for i in range(NUM_TRAIN_IMGS):
    idx = random.randint(0, len(imgs) - 1)
    train_imgs.append(imgs[idx])

    for row in csv_rows:
        if row[0] == imgs[idx]:
            train_ann.append(row)

    imgs.remove(imgs[idx])

for i in range(NUM_TEST_IMGS):
    idx = random.randint(0, len(imgs) - 1)
    test_imgs.append(imgs[idx])

    for row in csv_rows:
        if row[0] == imgs[idx]:
            test_ann.append(row)

    imgs.remove(imgs[idx])

for i in range(NUM_VAL_IMGS):
    idx = random.randint(0, len(imgs) - 1)
    val_imgs.append(imgs[idx])

    for row in csv_rows:
        if row[0] == imgs[idx]:
            val_ann.append(row)

    imgs.remove(imgs[idx])

with open(ANN_SAVE_PATH + "train_label.csv", 'w') as file:
    writer = csv.writer(file)
    writer.writerows(train_ann)

with open(ANN_SAVE_PATH + "test_label.csv", 'w') as file:
    writer = csv.writer(file)
    writer.writerows(test_ann)

with open(ANN_SAVE_PATH + "val_label.csv", 'w') as file:
    writer = csv.writer(file)
    writer.writerows(val_ann)

for file in train_imgs:
    copyfile(IMGS_PATH + file, IMG_SAVE_PATH + "train/" + file)

for file in test_imgs:
    copyfile(IMGS_PATH + file, IMG_SAVE_PATH + "test/" + file)

for file in val_imgs:
    copyfile(IMGS_PATH + file, IMG_SAVE_PATH + "val/" + file)

print(f'Subset sizes:\n'
      f'Train: {len(train_imgs)}\n'
      f'Test: {len(test_imgs)}\n'
      f'Val: {len(val_imgs)}')
