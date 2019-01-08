import time
import copy
import pandas as pd
from collections import defaultdict
import numpy as np
class2label={"DESERT":0,"MOUNTAIN":1,"OCEAN":2,"FARMLAND":3,"LAKE":4,"CITY":5,"UNKNOW":6}
label2class=["DESERT","MOUNTAIN","OCEAN","FARMLAND","LAKE","CITY","UNKNOW"]

result_ratios = defaultdict(lambda: 0)
mode="test"
result_ratios['resnet18'] = 0.08
result_ratios['se-resnet18'] = 0.04
result_ratios['resnet34'] = 0.08
result_ratios['resnet18_224'] = 0.2
result_ratios['resnet50_224'] = 0.6

assert sum(result_ratios.values()) == 1
def str2np(str):
    return np.array([float(x) for x in str.split(";")])
def np2str(arr):
    return ";".join(["%.16f" % x for x in arr])
for index, model in enumerate(result_ratios.keys()):
    print('ratio: %.3f, model: %s' % (result_ratios[model], model))
    result = pd.read_csv('lsz/test_csv/{}_result_b_stride1_{}_prob.csv'.format(model,mode),names=["probability"])
    # result = result.sort_values(by='filename').reset_index(drop=True)
    result['probability'] = result['probability'].apply(lambda x: str2np(x))
    print(result.head(3))

    if index == 0:
        ensembled_result = copy.deepcopy(result)
        ensembled_result['probability'] =0

    ensembled_result['probability'] = ensembled_result['probability'] + result['probability']*result_ratios[model]
    print(ensembled_result.head(3))

f = open('lsz/merge/{}_{}__{}_resnet18_50_224_resultb_108.csv'.format(result_ratios.keys()[0],result_ratios.keys()[1],result_ratios.keys()[2]), 'a')
for idx,probability in enumerate(ensembled_result["probability"].tolist()):
    one_hot = np.zeros(17, dtype=np.int8)
    label =np.argmax(probability)
    one_hot[int(label)] =1
    one_hot_str =",".join([str(x) for x in list(one_hot)])
    f.write('%s\n' % (one_hot_str))
f.close()
