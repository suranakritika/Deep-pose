import scipy.io as sc
import numpy as np
import glob
from os.path import basename as b
import re

joints = sc.loadmat("LSP_Dataset/joints.mat")
joints = joints['joints'].transpose(2,0,1)
joints = joints[:, :, :2]

N_test = int(len(joints)*0.1)
perm = np.random.permutation(int(len(joints)))[:N_test].tolist()

fp_train = open('train_joints.csv', 'w')
fp_test = open('test_joints.csv', 'w')
for img_fn in sorted(glob.glob('LSP_Dataset/images/*.jpg')):
    index = int(re.search('im([0-9]+)', b(img_fn)).groups()[0]) - 1
    str_j = [str(j) if j > 0 else '-1'
             for j in joints[index].flatten().tolist()]

    out_list = [b(img_fn)]
    out_list.extend(str_j)
    out_str = ','.join(out_list)

    if index in perm:
        print(out_str, file=fp_test)
    else:
        print(out_str, file=fp_train)
fp_train.close()
fp_test.close()