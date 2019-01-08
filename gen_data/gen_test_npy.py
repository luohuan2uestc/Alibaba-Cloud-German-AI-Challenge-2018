import numpy as np
import h5py

# filename = "/media/hszc/model/detao/data/LCZ42/round1_test_a_20181109.h5"
filename="/media/hszc/model/detao/data/LCZ42/round1_test_b_20190104.h5"
f = h5py.File(filename,'r')
print('Get the h5 file')

s1 = np.array(f['sen1'])
s2 = np.array(f['sen2'])
print(s1.shape)

x = []
for i in range(0,s1.shape[0]):
    temp1 = s1[i].flatten()
    temp2 = s2[i].flatten()
    temp = np.hstack((temp1,temp2))
    x.append(temp)
data = np.array(x)
print(data.shape)
np.save('round1_test_b_20190104.npy',data)