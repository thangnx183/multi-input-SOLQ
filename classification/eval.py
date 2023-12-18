import torchvision.transforms as transforms
import torch
import numpy as np
import torchvision
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from utils import progress_bar
import os

import timm

transform_test = transforms.Compose([
    transforms.Resize((448,448)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

data_path = '../coco_data/dent/20231208/'
device='cuda:0'


testset = torchvision.datasets.ImageFolder(
    root=data_path+'val', transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=16, shuffle=False, num_workers=2)

model = timm.create_model('eva02_base_patch14_448.mim_in22k_ft_in22k', pretrained=False,num_classes=2)

ckpt = torch.load('checkpoint/ckpt_dent.pth',map_location='cpu')
model.load_state_dict(ckpt['net'])
model.to(device)

correct = 0
total = 0
mc = [[0,0],[0,0]]

with torch.no_grad():
    for batch_idx, (inputs, targets) in enumerate(testloader):
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        # loss = criterion(outputs, targets)

        # test_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        tp00 = predicted[targets==0].eq(targets[targets==0]).sum().item()
        fp01 = predicted[predicted==0].shape[0] - tp00
        tp11 = predicted[targets==1].eq(targets[targets==1]).sum().item()
        fp10 = predicted[predicted==1].shape[0] - tp11
        
        mc[0][0] += tp00
        mc[0][1] += fp01
        mc[1][0] += fp10
        mc[1][1] += tp11
        print(targets,predicted)
        print(mc)


