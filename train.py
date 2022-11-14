import os
import torch
import time
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import torch.optim as optim
from torch.optim import lr_scheduler
from configargparse import ArgParser
import torch.utils.data as Data
from util.pytorchtools import EarlyStopping
from torch.autograd import Variable
from torchnet.meter import AverageValueMeter
from data_processing.division import get_division_len
from model.model import DI_ConvGRU
from model.crit import RmseLoss
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"


class subDataset(Data.Dataset):
    #初始化，定义数据内容和标签
    def __init__(self, Data_ERA5, Data_sm, Label):
        self.Data_ERA5 = Data_ERA5
        self.Data_sm = Data_sm
        self.Label = Label
    #返回数据集大小
    def __len__(self):
        return len(self.Data_sm)
    #得到数据内容和标签
    def __getitem__(self, index):
        data_ERA5 = self.Data_ERA5[index]
        data_sm = self.Data_sm[index]
        label = self.Label[index]
        return data_ERA5, data_sm, label

def train_valid_seq_split(ERA5_train, sm_data, label_train, batch_size, shuffle=True):
    dataset = subDataset(ERA5_train, sm_data, label_train)
    train_loader = Data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)
    return train_loader

def train(lr, train_loader, validation_loader, total_epoch, early_stopping):
    ERA5_train = np.load('./data/data_division0/data_model_lag3/' + 'ERA5_train.npy')

    sample, time_step, features, lons, lats = ERA5_train.shape
    model = DI_ConvGRU(input_size=(lons, lats), input_dim=2,
                        hidden_dim=[64, 64, 1], kernel_size=(3, 3),
                        num_layers=3)
    model.cuda()
    global_step = 1
    optimizer = optim.Adam(model.parameters(), lr=lr, eps=1e-3)
    loss_func = RmseLoss()
    loss_metrics = AverageValueMeter()
    #scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[i for i in range(0, 50, 10)][1:], gamma=0.05)
    ########## training set##########
    for epoch in range(total_epoch):
        epoch_loss = 0
        t0 = time.time()
        total_time = 0
        epoch_loss_validation = 0
        for step, (data1, data2, label) in tqdm(enumerate(train_loader)):
            optimizer.zero_grad()
            ERA5_train = torch.as_tensor(Variable(data1), dtype=torch.float).cuda()
            sm_train = torch.as_tensor(Variable(data2), dtype=torch.float).cuda()
            label_train = torch.as_tensor(Variable(label), dtype=torch.float).cuda()
            pred = model(ERA5_train, sm_train)
            train_loss = loss_func(pred, label_train).requires_grad_()
            train_loss.backward()
            optimizer.step()
            global_step = global_step + 1
            epoch_loss += train_loss.item()
            loss_metrics.add(train_loss.item())
            total_time += time.time() - t0
        print("epcho {}:loss {} time {:.2f}".format(epoch, epoch_loss/len(train_loader),time.time() - t0))
        # print("[epcho {}]:loss {}".format(epoch, loss_metrics.value()[0]))
        loss_metrics.reset()
        #scheduler.step()
        for step, (data1, data2, label) in tqdm(enumerate(validation_loader)):
            ERA5_validation = torch.as_tensor(Variable(data1), dtype=torch.float).cuda()
            sm_validation = torch.as_tensor(Variable(data2), dtype=torch.float).cuda()
            label_validation = torch.as_tensor(Variable(label), dtype=torch.float).cuda()
            pred = model(ERA5_validation, sm_validation)
            train_loss = loss_func(pred, label_validation).requires_grad_()
            epoch_loss_validation += train_loss.item()
        early_stopping(epoch_loss_validation, model)
        # 若满足 early stopping 要求
        if early_stopping.early_stop:
            print("Early stopping")
            # 结束模型训练
            break
    return model


def main(lr, total_epoch, batch_size, division_size, division_bias):
    lon = np.load('./data/lon.npy')
    lat = np.load('./data/lat.npy')
    width, height, widtha, heightb, widtha_down, heightb_rigint \
        = get_division_len(lon, lat, division_size, division_bias)

    count = 0
    division_len = (len(widtha) * len(heightb)) + (len(widtha_down) * len(heightb_rigint))
    for i in range(division_len):
        dir = r"./data/data_division" + str(count)
        dir1 = dir + '/data_model_lag3/'
        #train
        ERA5_train = np.load(dir1 + 'ERA5_train.npy')
        ERA5_train = ERA5_train[:,:,3,:,:]
        ERA5_train = ERA5_train[:,:, np.newaxis, :, :]
        static_train = np.load(dir1 + 'static_train.npy')
        # static_train = static_train[:,:,0:4,:,:]
        sm_train = np.load(dir1 + 'sm_train.npy')
        label_train = np.load(dir1 + 'label_train.npy')
        #features_train = np.concatenate((ERA5_train, static_train), axis=2)
        #validation
        ERA5_validation = np.load(dir1 + 'ERA5_validation.npy')
        ERA5_validation = ERA5_validation[:, :, 3, :, :]
        ERA5_validation = ERA5_validation[:, :, np.newaxis, :, :]
        static_validation = np.load(dir1 + 'static_validation.npy')
        # static_validation = static_validation[:, :, 0:4, :, :]
        #features_validation = np.concatenate((ERA5_validation, static_validation), axis=2)
        sm_validation = np.load(dir1 + 'sm_validation.npy')
        label_validation = np.load(dir1 + 'label_validation.npy')
        # 初始化 early_stopping 对象
        patience = 100  # 当验证集损失在连续20次训练周期中都没有得到降低时，停止模型训练，以防止模型过拟合
        early_stopping = EarlyStopping(patience, verbose=True)
        label_train2 = label_train.reshape(label_train.shape[0],
        label_train.shape[1]*label_train.shape[2]*label_train.shape[3])
        if np.min(label_train2) != np.nan:
            # TODO: transform data for model
            #####################数据分块存放在GPU里，进行训练
            train_loader = train_valid_seq_split(ERA5_train, sm_train, label_train, batch_size, shuffle=True)
            validation_loader = train_valid_seq_split(ERA5_validation, sm_validation,
                                                      label_validation, batch_size, shuffle=True)

            # TODO: train LSTM model
            model = train(lr, train_loader, validation_loader, total_epoch, early_stopping)
            torch.save(model.state_dict(), dir1 + 'model_tp' + '.pth')
            print(dir, 'dir')
            print('finished training model' + str(count) + 'and' + str(division_len - count) + 'left')
        else:
            print('see reggion' + str(count) + 'and' + str(division_len - count) + 'left')
        count = count + 1


if __name__ == '__main__':
    p = ArgParser()
    p.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    p.add_argument('--total_epoch', type=int, default=100, help='total epochs for training the model')
    p.add_argument('--batch_size', type=int, default=16, help='batch_size')
    p.add_argument('--division_size', type=int, default=50, help='division size of small input for training the model')
    p.add_argument('--division_bias', type=int, default=8, help='division bias of small input for training the model')
    args = p.parse_args()

    main(
        lr=args.lr,
        total_epoch=args.total_epoch,
        batch_size=args.batch_size,
        division_size=args.division_size,
        division_bias=args.division_bias
    )
