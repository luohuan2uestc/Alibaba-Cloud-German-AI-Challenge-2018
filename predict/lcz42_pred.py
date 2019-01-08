#coding=utf-8
import os
import numpy as np
import pandas as pd
from dataset.lcz42_dataset import dataset, collate_fn
import torch
from torch.nn import CrossEntropyLoss
import torch.utils.data as torchdata
from torchvision import datasets, models, transforms
from models.resnet_lcs42 import resnet50,resnet18_lcs,resnet34_lcs
from models.senet_lcs42 import se_resnet18_lcs42,se_resnet34_lcs42
from sklearn.model_selection import train_test_split
from torch.autograd import Variable
from math import ceil
from  torch.nn.functional import softmax,log_softmax
from dataset.data_aug import *
import glob
os.environ["CUDA_VISIBLE_DEVICES"] = "0,2,3"
mode ="test"
model_name ="resnet101_224"   # "resnet18",se-resnet18,resnet34,se-resnet34,resnet18_224,resnet50_224,resnet101_224
model_version ="stride1"   #onlyvaldata,nomaxpool

if mode=="test":
    # test_data = "/home/detao/Desktop/pytorch_classification/round1_test_a_20181109.npy"
    test_data = "../gen_data/round1_test_b_20190104.npy"
    fid_data = np.load(test_data)
    fid_data = fid_data[:, 8192:]
else:
    val_data="/home/detao/Desktop/pytorch_classification/data_shuffle_10000.npy"
    # val_data="/home/detao/Desktop/pytorch_classification/vali_data.npy"

    val_data = np.load(val_data)
    fid_data = val_data[100000:, 8192:]
    # fid_data = val_data[21000:, 8192:]
test_mode =True if mode=="test" else False
data_transforms = {
    'train': Compose([
        Resize((224, 224)),
        FixRandomRotate(),
        RandomHflip(),
        RandomVflip(),
    ]),
    'val': Compose([
        Resize((224, 224)),
    ]),
}
data_transforms =data_transforms["val"] if model_name=="resnet18_224" or model_name=="resnet50_224" or model_name=="resnet101_224"  else None
data_set = {}
data_set['test'] = dataset(fid_data,transforms=data_transforms,test_mode=test_mode)
data_loader = {}
data_loader['test'] = torchdata.DataLoader(data_set['test'], batch_size=8, num_workers=4,
                                           shuffle=False, pin_memory=True, collate_fn=collate_fn)
if model_name=="resnet18":
    # resume = '/media/hszc/model/detao/models/lcz42/resnet18_lcs_256/best_weigths_[0.0001].pth'
    resume = "/media/hszc/model/detao/models/lcz42/resnet18_lcs42_stride1/best_weigths_[1.0000000000000003e-05].pth"
    # resume="/media/hszc/model/detao/models/lcz42/resnet18_lcs42_stride1_nomaxpool/best_weigths_[1.0000000000000003e-05].pth"
    # resume="/media/hszc/model/detao/models/lcz42/resnet18_lcs42_stride1_onlyvaldata/best_weigths_[1.0000000000000003e-05].pth"
    model = resnet18_lcs()
elif model_name=="se-resnet18":
    # resume ="/media/hszc/model/detao/models/lcz42/se_resnet18_lcs42/best_weigths_[1.0000000000000003e-05].pth"
    # resume ="model_weights/se_resnet18_9254/best_weigths_[1.0000000000000003e-05].pth"
    resume="/media/hszc/model/detao/models/lcz42/se_resnet18_lcs42_dataaug/best_weigths_[1.0000000000000003e-05].pth"
    model= se_resnet18_lcs42()
elif model_name == "resnet34":
    resume="/media/hszc/model/detao/models/lcz42/resnet34_lcs42_stride1/best_weigths_[0.0001].pth"
    model= resnet34_lcs()
elif model_name == "se-resnet34":
    resume="/media/hszc/model/detao/models/lcz42/dla34_lcs/best_weigths_[1.0000000000000003e-05].pth"
    model= se_resnet34_lcs42()
elif model_name =="resnet18_224":
    from torchvision.models import resnet18
    resume="/media/hszc/model/detao/models/lcz42/resnet18_224_input/best_weigths_[1.0000000000000003e-05].pth"
    model=resnet18(pretrained=True)
    model.conv1 = torch.nn.Conv2d(10, 64, kernel_size=7, stride=1, padding=3,
                               bias=False)
    model.avgpool = torch.nn.AdaptiveAvgPool2d(output_size=1)
    model.fc = torch.nn.Linear(model.fc.in_features,17)
    model = torch.nn.DataParallel(model)
elif model_name =="resnet50_224":
    from torchvision.models import resnet50
    resume="/media/hszc/model/detao/models/lcz42/resnet50_224_input/best_weigths_[0.0001].pth"
    model=resnet50(pretrained=True)
    model.conv1 = torch.nn.Conv2d(10, 64, kernel_size=7, stride=1, padding=3,
                               bias=False)
    model.avgpool = torch.nn.AdaptiveAvgPool2d(output_size=1)
    model.fc = torch.nn.Linear(model.fc.in_features,17)
    model = torch.nn.DataParallel(model)
elif model_name =="resnet101_224":
    from torchvision.models import resnet101
    resume="/media/hszc/model/detao/models/lcz42/resnet101_224_input/best_weigths_[0.0001].pth"
    model=resnet101(pretrained=True)
    model.conv1 = torch.nn.Conv2d(10, 64, kernel_size=7, stride=1, padding=3,
                               bias=False)
    model.avgpool = torch.nn.AdaptiveAvgPool2d(output_size=1)
    model.fc = torch.nn.Linear(model.fc.in_features,17)
    model = torch.nn.DataParallel(model)
# from models.dla_lcs42 import dla34_lcs
# resume="/media/hszc/model/detao/models/lcz42/dla34_lcs/best_weigths_[0.00001]_onlyvaldata.pth"
# model =dla34_lcs()
print('resuming finetune from %s'%resume)
model.load_state_dict(torch.load(resume))

model = model.cuda()
model.eval()

criterion = CrossEntropyLoss()

if not os.path.exists('./lsz/{}'.format(mode)):
    os.makedirs('./lsz/{}'.format(mode))

test_size = ceil(len(data_set['test']) / data_loader['test'].batch_size)
test_preds = np.zeros((len(data_set['test'])), dtype=np.float32)
true_label = np.zeros((len(data_set['test'])), dtype=np.int)
test_scores = np.zeros((len(data_set['test']), 17), dtype=np.float32)

idx = 0
test_loss = 0
test_corrects = 0
for batch_cnt_test, data_test in enumerate(data_loader['test']):
    # print data
    print("{0}/{1}".format(batch_cnt_test, int(test_size)))
    inputs, labels = data_test
    inputs = Variable(inputs.cuda())
    labels = Variable(torch.from_numpy(np.array(labels)).long().cuda())
    # forward
    outputs = model(inputs)

    # statistics
    if isinstance(outputs, list):
        loss = criterion(outputs[0], labels)
        loss += criterion(outputs[1], labels)
        outputs = (outputs[0]+outputs[1])/2
    else:
        loss = criterion(outputs, labels)
    _, preds = torch.max(outputs, 1)

    scores = softmax(outputs,1)
    print(scores.size())

    test_loss += loss.data[0]
    batch_corrects = torch.sum((preds == labels)).data[0]
    test_corrects += batch_corrects
    test_preds[idx:(idx + labels.size(0))] = preds
    true_label[idx:(idx + labels.size(0))] = labels.data.cpu().numpy()
    test_scores[idx:(idx + labels.size(0))] = scores.data.cpu().numpy()

    # statistics
    idx += labels.size(0)
test_loss = test_loss / test_size
print(test_corrects)
test_acc = 1.0 * test_corrects / len(data_set['test'])
print('test-loss: %.4f ||test-acc@1: %.4f'
      % (test_loss, test_acc))
def np2str(arr):
    return ";".join(["%.8f" % x for x in arr])
def str2np(str):
    return np.array([x for x in str.split(";")])
test_pred = pd.DataFrame(list(test_preds),columns=["label"])
test_pred['prob'] = list(test_scores)
test_pred['prob'] = test_pred['prob'].apply(lambda x: np2str(x))
test_pred[["prob"]].to_csv('lsz/{}/{}_result_b_{}_{}_prob.csv'.format(mode,model_name,model_version,mode), sep=",",
                                                 header=None, index=False)
f = open('lsz/{}/{}_result_b_{}_{}.csv'.format(mode,model_name,model_version,mode), 'a')
for idx,label in enumerate(test_preds):
    one_hot = np.zeros(17, dtype=np.int8)
    one_hot[int(label)] =1
    one_hot_str =",".join([str(x) for x in list(one_hot)])
    f.write('%s\n' % (one_hot_str))
f.close()

# 23463
# test-loss: 0.0983 ||test-acc@1: 0.9728