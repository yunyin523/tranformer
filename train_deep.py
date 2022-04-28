from functools import reduce
from unittest import result
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

import numpy as np
from model.utnet import UTNet, UTNet_Encoderonly

from dataset_domain import CMRDataset

from torch.utils import data
from losses import DiceLoss
from utils.utils import *
from utils import metrics
from optparse import OptionParser
import SimpleITK as sitk

from torch.utils.tensorboard import SummaryWriter
import time
import math
import os
import sys
import pdb
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
DEBUG = False

def train_net(net, options):
    
    data_path = options.data_path

    trainset = CMRDataset(data_path, mode='train', domain=options.domain, debug=DEBUG, scale=options.scale, rotate=options.rotate, crop_size=options.crop_size)
    trainLoader = data.DataLoader(trainset, batch_size=options.batch_size, shuffle=True, num_workers=16)
    testset = CMRDataset(data_path, mode='test', domain='A', debug=DEBUG, crop_size=options.crop_size)
    testLoader_A = data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=2)
    testset = CMRDataset(data_path, mode='test', domain='B', debug=DEBUG, crop_size=options.crop_size)
    testLoader_B = data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=2)
    testset = CMRDataset(data_path, mode='test', domain='C', debug=DEBUG, crop_size=options.crop_size)
    testLoader_C = data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=2)

    writer = SummaryWriter(options.log_path + options.unique_name)

    optimizer = optim.SGD(net.parameters(), lr=options.lr, momentum=0.9, weight_decay=options.weight_decay)

    criterion = nn.CrossEntropyLoss(weight=torch.tensor(options.weight).cuda())
    criterion_dl = DiceLoss()

    checkpoint = torch.load('%s%s/CP%d.pth'%(options.cp_path, options.unique_name, 80))

    net.load_state_dict(checkpoint['net'])
    optimizer.load_state_dict(checkpoint['optimizer'])

    # start_epoch = 0
    start_epoch = checkpoint['epoch']

    for epoch in range(start_epoch, options.epochs):
        print('Starting epoch {}/{}'.format(epoch+1, options.epochs))
        epoch_loss = 0

        exp_scheduler = exp_lr_scheduler_with_warmup(optimizer, init_lr=options.lr, epoch=epoch, warmup_epoch=5, max_epoch=options.epochs)

        print('current learning rate:', exp_scheduler)

        for i, (img, label) in enumerate(trainLoader, 0):
            
            img = img.cuda()
            label = label.cuda()

            end = time.time()
            net.train()

            optimizer.zero_grad()
            result = net(img)

            loss = 0
            if isinstance(result, tuple) or isinstance(result, list):
                for j in range(len(result)):
                    loss += options.aux_weight[j] * (criterion(result[j], label.squeeze(1)) + criterion_dl(result[j], label))
            else:
                loss = criterion(result, label.squeeze(1)) + criterion_dl(result, label)


            loss.backward()
            optimizer.step()


            epoch_loss += loss.item()
            batch_time = time.time() - end
            print('batch loss: %.5f, batch_time:%.5f'%(loss.item(), batch_time))
        print('[epoch %d] epoch loss: %.5f'%(epoch+1, epoch_loss/(i+1)))

        writer.add_scalar('Train/Loss', epoch_loss/(i+1), epoch+1)
        writer.add_scalar('LR', exp_scheduler, epoch+1)

        if os.path.isdir('%s%s/'%(options.cp_path, options.unique_name)):
            pass
        else:
            os.mkdir('%s%s/'%(options.cp_path, options.unique_name))
        if epoch % 20 == 0 or epoch > options.epochs-10:
            state = {'net':net.state_dict(), 'optimizer':optimizer.state_dict(), 'epoch':epoch}
            torch.save(state, '%s%s/CP%d.pth'%(options.cp_path, options.unique_name, epoch))
        if (epoch + 1) > 90 or epoch > (options.epochs + 1) % 10 == 0:
            validation(net, testLoader_A, options, "Cirrus", epoch)
            validation(net, testLoader_B, options, "Spectralis", epoch)
            validation(net, testLoader_C, options, "Topcon", epoch)

def validation(net, test_loader, options, domain, epoch):
    net.eval()
    class_name = ["space", "IRF", "SRF", "PED"]
    writer = SummaryWriter(options.log_path + options.unique_name)
    print("Testing epoch {}".format(epoch))
    dsc1 = []
    dsc2 = []
    dsc3 = []
    dsc_list = [0,0,0]
    dsc_std = [0,0,0]
    count = 0
    for i, (img, label) in enumerate(test_loader, 0):
        print("epoch "+str(epoch) + " " +domain + " " + str(count))
        count += 1
        img = img.cuda()
        label = label.cuda()
        result = net(img)
        label_narray = label.detach().cpu().numpy()
        dsc = [0,0,0,0]
        if isinstance(result, tuple) or isinstance(result, list):
            for j in range(len(result)):
                result_narray = result[j].detach().cpu().numpy()
                dsc_ = cal_dsc(options, result_narray, label_narray)
                for k in range(options.num_class):
                    dsc[k] += dsc_[k]
            for j in range(options.num_class):
                dsc[j] /= len(result)
        else:
            result_narray = result.detach().numpy()
            dsc_ = cal_dsc(options, result_narray, label_narray)
            for j in range(options.num_class):
                dsc[j] += dsc_[j]
        dsc1.append(dsc[1])
        dsc2.append(dsc[2])
        dsc3.append(dsc[3])
    dsc_list[0] = np.mean(dsc1)
    dsc_list[1] = np.mean(dsc2)
    dsc_list[2] = np.mean(dsc3)
    dsc_std[0] = np.std(dsc1)
    dsc_std[1] = np.std(dsc2)
    dsc_std[2] = np.std(dsc3)
    dsc_all = dsc1 + dsc2 + dsc3
    DSC_mean = np.mean(dsc_all)
    DSC_std = np.std(dsc_all)
    print('[epoch {}] TEST {} DSC mean : {:.5f}, DSC std : {:.5f}'.format(epoch, domain, DSC_mean, DSC_std))
    writer.add_scalar('Test/{}/DSC/ALL/mean'.format(domain), DSC_mean, epoch)
    writer.add_scalar('Test/{}/DSC/ALL/std'.format(domain), DSC_std, epoch)
    for i in range(1, options.num_class):
        writer.add_scalar('Test/{}/DSC/{}/mean'.format(domain, class_name[i]), dsc_list[i - 1], epoch)
        writer.add_scalar('Test/{}/DSC/{}/std'.format(domain, class_name[i]), dsc_std[i - 1], epoch)

def cal_dsc(options, result, label):
    pred = np.zeros(label.shape)
    for i in range(result.shape[0]):
        for j in range(result.shape[2]):
            for k in range(result.shape[3]):
                max_t = result[i][0][j][k]
                index = 0
                for t in range(0, result.shape[1]):
                    if(result[i][t][j][k] > max_t):
                        max_t = result[i][t][j][k]
                        index = t
                pred[i][0][j][k] = index
    flat_pred = np.ndarray.flatten(pred)
    flat_label = np.ndarray.flatten(label)
    classes_dsc = [0]
    for i in range(1, options.num_class):
        A = flat_label == i
        B = flat_pred == i
        inter = np.sum(A * B)
        union = np.sum(A) + np.sum(B)
        classes_dsc.append((2 * inter + 0.00001)/ (union + 0.00001))
    print(classes_dsc)
    return classes_dsc

if __name__ == '__main__':
    parser = OptionParser()
    def get_comma_separated_int_args(option, opt, value, parser):
        value_list = value.split(',')
        value_list = [int(i) for i in value_list]
        setattr(parser.values, option.dest, value_list)

    parser.add_option('-e', '--epochs', dest='epochs', default=150, type='int', help='number of epochs')
    parser.add_option('-b', '--batch_size', dest='batch_size', default=16, type='int', help='batch size')
    parser.add_option('-l', '--learning-rate', dest='lr', default=0.05, type='float', help='learning rate')
    parser.add_option('-c', '--resume', type='str', dest='load', default=False, help='load pretrained model')
    parser.add_option('-p', '--checkpoint-path', type='str', dest='cp_path', default='/home/vislab/yueyang/graduation_project/UTNet/dataset/checkpoint/', help='checkpoint path')
    parser.add_option('--data_path', type='str', dest='data_path', default='/home/vislab/yueyang/graduation_project/UTNet/dataset/', help='dataset path')

    parser.add_option('-o', '--log-path', type='str', dest='log_path', default='./log/', help='log path')
    parser.add_option('-m', type='str', dest='model', default='UTNet', help='use which model')
    parser.add_option('--num_class', type='int', dest='num_class', default=4, help='number of segmentation classes')
    parser.add_option('--base_chan', type='int', dest='base_chan', default=32, help='number of channels of first expansion in UNet')
    parser.add_option('-u', '--unique_name', type='str', dest='unique_name', default='test', help='unique experiment name')
    parser.add_option('--rlt', type='float', dest='rlt', default=1, help='relation between CE/FL and dice')
    parser.add_option('--weight', type='float', dest='weight',
                      default=[0.5,1,1,1] , help='weight each class in loss function')
    parser.add_option('--weight_decay', type='float', dest='weight_decay',
                      default=0.0001)
    parser.add_option('--scale', type='float', dest='scale', default=0.30)
    parser.add_option('--rotate', type='float', dest='rotate', default=180)
    parser.add_option('--crop_size', type='int', dest='crop_size', default=256)
    parser.add_option('--domain', type='str', dest='domain', default='ABC')
    parser.add_option('--aux_weight', type='float', dest='aux_weight', default=[1, 0.4, 0.2, 0.1])
    parser.add_option('--reduce_size', dest='reduce_size', default=8, type='int')
    parser.add_option('--block_list', dest='block_list', default='234', type='str')
    parser.add_option('--num_blocks', dest='num_blocks', default=[1,1,1,1], type='string', action='callback', callback=get_comma_separated_int_args)
    parser.add_option('--aux_loss', dest='aux_loss', action='store_true', help='using aux loss for deep supervision')

    
    parser.add_option('--gpu', type='str', dest='gpu', default='0')
    options, args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = options.gpu

    print('Using model:', options.model)

    if options.model == 'UTNet':
        net = UTNet(1, options.base_chan, options.num_class, reduce_size=options.reduce_size, block_list=options.block_list, num_blocks=options.num_blocks, num_heads=[4,4,4,4], projection='interp', attn_drop=0.1, proj_drop=0.1, rel_pos=True, aux_loss=options.aux_loss, maxpool=True)
    else:
        raise NotImplementedError(options.model + " has not been implemented")
    if options.load:
        net.load_state_dict(torch.load(options.load))
        print('Model loaded from {}'.format(options.load))
    
    param_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    
    print(net)
    print(param_num)
    
    net.cuda()
    
    train_net(net, options)

    print('done')

    sys.exit(0)
