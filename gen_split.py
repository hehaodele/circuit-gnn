import os
import numpy as np

num_block = [3, 4, 5, 6]
train_num_block = [3, 4, 5]

folders = os.listdir('data')

train_list = []
valid_list = []
test_list = []

for num in num_block:

    block_folders = sorted([*filter(lambda x: x.startswith(f'num{num}'), folders)])

    if num in train_num_block:

        for fd in block_folders:
            fns = os.listdir('data' + '/' + fd)
            size = len(fns)
            train_size = int(size * 0.8)
            valid_size = int(size * 0.1)
            test_size = size - train_size - valid_size
            train_list += [fd + '/' + fn for fn in fns[:train_size]]
            valid_list += [fd + '/' + fn for fn in fns[train_size:train_size + valid_size]]
            test_list += [fd + '/' + fn for fn in fns[train_size + valid_size:]]

    else:
        for fd in block_folders:
            fns = os.listdir(block_folders)
            test_list += [fd + '/' + fn for fn in fns]

with open("./data/train_list.txt", "w") as f:
    for x in train_list:
        f.write(x + '\n')
with open("./data/valid_list.txt", "w") as f:
    for x in valid_list:
        f.write(x + '\n')
with open("./data/test_list.txt", "w") as f:
    for x in test_list:
        f.write(x + '\n')
