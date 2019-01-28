#coding=utf-8
import os
from dataset.lcz42_dataset import collate_fn, dataset
import torch
import torch.utils.data as torchdata
from models.resnet_lcs42 import resnet18_lcs,resnet34_lcs
from models.senet_lcs42 import se_resnet18_lcs42,se_resnet34_lcs42
from models.dla_lcs42 import dla34_lcs
import torch.optim as optim
from torch.optim import lr_scheduler
from utils.train_util import train, trainlog
from  torch.nn import CrossEntropyLoss
import logging
from dataset.data_aug import *
import sys
import argparse
from torchvision.models import resnet18,resnet50,resnet101

reload(sys)
sys.setdefaultencoding('utf8')

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=128, help='size of each image batch')

parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')
parser.add_argument('--data_path', type=str, default= "/home/detao/Desktop/pytorch_classification/data_shuffle_10000.npy", help='whether to img root')
parser.add_argument('--checkpoint_dir', type=str, default='/media/hszc/model/detao/models/lcz42/resnet101_128', help='directory where model checkpoints are saved')
parser.add_argument('--cuda_device', type=str, default="0,2,3", help='whether to use cuda if available')
parser.add_argument('--net', dest='net',type=str, default='resnet101',help='resnet101,resnet50')
# parser.add_argument('--resume', type=str, default="/media/hszc/model/detao/models/lcz42/dla34_lcs/best_weigths_[0.001].pth", help='path to resume weights file')
# parser.add_argument('--resume', type=str, default="/media/hszc/model/detao/models/lcz42/resnet50_val_train/best_weigths_[0.0001].pth", help='path to resume weights file')
parser.add_argument('--resume', type=str, default=None, help='path to resume weights file')

parser.add_argument('--epochs', type=int, default=90, help='number of epochs')
parser.add_argument('--start_epoch', type=int, default=0, help='number of start epoch')

parser.add_argument('--save_checkpoint_val_interval', type=int, default=2000, help='interval between saving model weights')
parser.add_argument('--print_interval', type=int, default=30, help='interval between print log')

parser.add_argument('--n_cpu', type=int, default=4, help='number of cpu threads to use during batch generation')
opt = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = opt.cuda_device
if __name__ == '__main__':
    import numpy as np
    # # saving dir
    save_dir = opt.checkpoint_dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    logfile = '%s/trainlog.log' % save_dir
    trainlog(logfile)
    data_all = np.load(opt.data_path)
    print(data_all.shape)
    train_data = data_all[:100000, 8192:]
    val_data = data_all[100000:, 8192:]
    # train_data = np.load("add_data_10000.npy")
    # val_data = np.load("vali_data.npy")
    # train_data = train_data[:, 8192:]
    # val_data = val_data[:, 8192:]
    print('The shape of train is ', train_data.shape)
    print('The shape of vali is ', val_data.shape)
    val_label =val_data[:,-17:]
    jl = np.sum(val_label,axis=0)
    print("val data",jl)
    data_transforms = {
        'train': Compose([
            Resize((224,224)),
            FixRandomRotate(),
            RandomHflip(),
            RandomVflip(),
        ]),
        'val': Compose([
            Resize((224, 224)),
        ]),
    }
    data_set = {}
    data_set['train'] = dataset(train_data, transforms=data_transforms["train"])
    data_set['val'] = dataset(val_data, transforms=data_transforms["val"])

    dataloader = {}
    dataloader['train']=torch.utils.data.DataLoader(data_set['train'], batch_size=opt.batch_size,
                                                   shuffle=True, num_workers=2*opt.n_cpu,collate_fn=collate_fn)
    dataloader['val']=torch.utils.data.DataLoader(data_set['val'], batch_size=16,
                                                   shuffle=False, num_workers=opt.n_cpu,collate_fn=collate_fn)
    '''model'''
    # if opt.net == "resnet50":
    #     model =resnet18(pretrained=True)
    #     model.conv1 = torch.nn.Conv2d(10, 64, kernel_size=3, stride=1, padding=3,
    #                            bias=False)
    #     model.avgpool = torch.nn.AdaptiveAvgPool2d(output_size=1)
    #     model.fc = torch.nn.Linear(model.fc.in_features,17)
    # elif opt.net == "resnet101":
    #     model =resnet101(pretrained=False)
    #     model.conv1 = torch.nn.Conv2d(10, 64, kernel_size=7, stride=1, padding=3,
    #                            bias=False)
    #     model.avgpool = torch.nn.AdaptiveAvgPool2d(output_size=1)
    #     model.fc = torch.nn.Linear(model.fc.in_features,17)
    # model =resnet18_lcs()
    # model =resnet34_lcs()
    # model=se_resnet34_lcs42()
    # model=resnet18(pretrained=True)
    # model=resnet50(pretrained=True)
    model=resnet101(pretrained=True)
    model.conv1 = torch.nn.Conv2d(10, 64, kernel_size=7, stride=1, padding=3,
                               bias=False)
    model.avgpool = torch.nn.AdaptiveAvgPool2d(output_size=1)
    model.fc = torch.nn.Linear(model.fc.in_features,17)
    model = torch.nn.DataParallel(model)

    # model =se_resnet18_lcs42()
    if opt.resume:
        model.eval()
        logging.info('resuming finetune from %s' % opt.resume)
        try:
            model.load_state_dict(torch.load(opt.resume))
        except KeyError:
            model = torch.nn.DataParallel(model)
            model.load_state_dict(torch.load(opt.resume))
    model = model.cuda()

    optimizer = optim.SGD(model.parameters(), lr=opt.learning_rate, momentum=0.9, weight_decay=1e-5)
    criterion = CrossEntropyLoss()
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.1)

    train(model,
          epoch_num=opt.epochs,
          start_epoch=opt.start_epoch,
          optimizer=optimizer,
          criterion=criterion,
          exp_lr_scheduler=exp_lr_scheduler,
          data_set=data_set,
          data_loader=dataloader,
          save_dir=save_dir,
          print_inter=opt.print_interval,
          val_inter=opt.save_checkpoint_val_interval)