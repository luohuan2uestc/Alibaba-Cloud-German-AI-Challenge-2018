
import h5py
import numpy as np
import matplotlib.pyplot as plt

filename_train = '/media/hszc/model/detao/data/LCZ42/training.h5'
f = h5py.File(filename_train,'r')

s1_train = f['sen1']
s2_train = f['sen2']
label_train = f['label']

filename_vali = '/media/hszc/model/detao/data/LCZ42/validation.h5'
f = h5py.File(filename_vali,'r')
s1_vali = f['sen1']
s2_vali = f['sen2']
label_vali = f['label']

sum_train = np.sum(label_train, axis=0)
sum_vali = np.sum(label_vali, axis=0)
print(sum_train)
print(sum_vali)

count = []
num = 8000
for i in range(label_train.shape[1]):
    if sum_vali[i] >= num:
        count.append(0)
    else:
        min_num = min(num,sum_train[i]+sum_vali[i])
        count.append(min_num-sum_vali[i])

count = np.array(count)
count = count.astype(np.int32)
print(count,sum(count))

didgit_label = np.argmax(label_train, 1)

ID = []
for index in range(17):
    c = 0
    for i in range(didgit_label.shape[0]):
        if c == count[index]:
            break
        if(didgit_label[i] == index):
            ID.append(i)
            c += 1
print(len(ID))
ID = sorted(ID)
add_label = label_train[ID,:]
xx = list(range(17))
sum_add_label = np.sum(add_label,axis = 0)
add_1 = s1_train[ID]
add_2 = s2_train[ID]

add_1 = np.array(add_1)
add_2 = np.array(add_2)
y = np.array(add_label)

x = []
for i in range(0,add_1.shape[0]):
    temp1 = add_1[i].flatten()
    temp2 = add_2[i].flatten()
    temp = np.hstack((temp1,temp2))
    x.append(temp)
x = np.array(x)

add_data = np.hstack((x,y))
print(add_data.shape)
np.save("add_data_10000.npy",add_data)