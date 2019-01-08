import numpy as np
import matplotlib.pyplot as plt
import random
vali_data = np.load('vali_data.npy')
add_data = np.load('add_data_10000.npy')

data = np.vstack((vali_data,add_data))
print(data.shape)

xx = list(range(17))
label = data[:,-17:]
sum_label = np.sum(label,axis = 0)
plt.bar(xx, sum_label,width = 0.5, color = "cornflowerblue")
my_x_ticks = np.arange(0, 17, 1)
#my_y_ticks = np.arange(0, 4000, 500)
plt.xticks(my_x_ticks)
#plt.yticks(my_y_ticks)
plt.xlim((-1, 17))
#plt.ylim((0, 3500))
plt.show()

data_shuffle = np.zeros(data.shape)
rand = list(range(0,data.shape[0]))
random.shuffle(rand)
for i in range(data.shape[0]):
    data_shuffle[i] = data[rand[i]]
batch_size = 1000
sample = random.sample(list(range(0,data.shape[0])),batch_size)
sample = sorted(sample)

batch = data_shuffle[sample,:]

x = batch[:,:-17]
y = batch[:,-17:]

xx = list(range(17))
sum_label = np.sum(y,axis = 0)
plt.bar(xx, sum_label,width = 0.5, color = "cornflowerblue")
my_x_ticks = np.arange(0, 17, 1)
#my_y_ticks = np.arange(0, 4000, 500)
plt.xticks(my_x_ticks)
#plt.yticks(my_y_ticks)
plt.xlim((-1, 17))
#plt.ylim((0, 3500))
plt.show()
np.save('data_shuffle_10000.npy',data_shuffle)
