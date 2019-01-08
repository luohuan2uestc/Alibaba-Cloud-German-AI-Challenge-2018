import numpy as np
import h5py

filename = '/media/hszc/model/detao/data/LCZ42/validation.h5'
f = h5py.File(filename,'r')
print('Get the h5 file')

s1 = np.array(f['sen1'])
s2 = np.array(f['sen2'])
y = np.array(f['label'])

x = []
for i in range(0,s1.shape[0]):
    temp1 = s1[i].flatten()
    temp2 = s2[i].flatten()
    temp = np.hstack((temp1,temp2))
    x.append(temp)
x = np.array(x)

data = np.hstack((x,y))
np.save('vali_data.npy',data)