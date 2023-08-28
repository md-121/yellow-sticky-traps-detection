import os
from random import randint
from shutil import copyfile

mr_imgs = []
for file in os.listdir("MR/"):
    if file.endswith(".png"):
        mr_imgs.append(file)

nc_imgs = []
for file in os.listdir("NC/"):
    if file.endswith(".png"):
        nc_imgs.append(file)

wf_imgs = []
for file in os.listdir("WF/"):
    if file.endswith(".png"):
        wf_imgs.append(file)

cnt_train_mr, cnt_train_nc, cnt_train_wf = 1132, 482, 4066
cnt_test_mr, cnt_test_nc, cnt_test_wf = 324, 137, 1162
cnt_val_mr, cnt_val_nc, cnt_val_wf = 162, 69, 580

train_mr, train_nc, train_wf = [], [], []
test_mr, test_nc, test_wf = [], [], []
val_mr, val_nc, val_wf = [], [], []

# train
while len(train_mr) != cnt_train_mr:
    i = randint(0, len(mr_imgs) - 1)
    train_mr.append(mr_imgs.pop(i))

while len(train_nc) != cnt_train_nc:
    i = randint(0, len(nc_imgs) - 1)
    train_nc.append(nc_imgs.pop(i))

while len(train_wf) != cnt_train_wf:
    i = randint(0, len(wf_imgs) - 1)
    train_wf.append(wf_imgs.pop(i))

# test
while len(test_mr) != cnt_test_mr:
    i = randint(0, len(mr_imgs) - 1)
    test_mr.append(mr_imgs.pop(i))

while len(test_nc) != cnt_test_nc:
    i = randint(0, len(nc_imgs) - 1)
    test_nc.append(nc_imgs.pop(i))

while len(test_wf) != cnt_test_wf:
    i = randint(0, len(wf_imgs) - 1)
    test_wf.append(wf_imgs.pop(i))

# val
while len(val_mr) != cnt_val_mr:
    i = randint(0, len(mr_imgs) - 1)
    val_mr.append(mr_imgs.pop(i))

while len(val_nc) != cnt_val_nc:
    i = randint(0, len(nc_imgs) - 1)
    val_nc.append(nc_imgs.pop(i))

while len(val_wf) != cnt_val_wf:
    i = randint(0, len(wf_imgs) - 1)
    val_wf.append(wf_imgs.pop(i))


# copy files
for file in train_mr:
    copyfile("MR/" + file, "train/MR/" + file)

for file in train_nc:
    copyfile("NC/" + file, "train/NC/" + file)

for file in train_wf:
    copyfile("WF/" + file, "train/WF/" + file)

# test
for file in test_mr:
    copyfile("MR/" + file, "test/MR/" + file)

for file in test_nc:
    copyfile("NC/" + file, "test/NC/" + file)

for file in test_wf:
    copyfile("WF/" + file, "test/WF/" + file)

# val
for file in val_mr:
    copyfile("MR/" + file, "val/MR/" + file)

for file in val_nc:
    copyfile("NC/" + file, "val/NC/" + file)

for file in val_wf:
    copyfile("WF/" + file, "val/WF/" + file)
