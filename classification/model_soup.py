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

model = timm.create_model(
    'eva02_base_patch14_448.mim_in22k_ft_in22k', pretrained=False, num_classes=2)

transform_test = transforms.Compose([
    transforms.Resize((448, 448)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

data_path = '../coco_data/dent/20231208/'
damage = 'dent'
device = 'cuda:1'


testset = torchvision.datasets.ImageFolder(
    root=data_path+'val', transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=20, shuffle=False, num_workers=2)


def test(model):
    # global best_acc
    # global best_loss
    model.eval()
    model.to(device)
    test_loss = 0
    correct = 0
    total = 0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            # val_loss = test_loss/(batch_idx+1)

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    return correct/total


def load_model_state_dict(model_path):
    ckpt = torch.load(model_path, map_location='cpu')
    return ckpt['net']


def uniform_soup(model_list):
    soup = {}
    NUM_MODELS = len(model_list)

    for idx, model_path in enumerate(model_list):
        sd = load_model_state_dict(model_path)
        if idx == 0:
            soup = {k: v/NUM_MODELS for k, v in sd.items()}
        else:
            soup = {k: v/NUM_MODELS+soup[k] for k, v in sd.items()}

    return soup


def greedy_soup(model_list):
    eval = {}
    for idx, model_path in enumerate(model_list):
        print('eval gradient : ', model_path)
        model.load_state_dict(load_model_state_dict(model_path))
        eval[model_path] = test(model)

    eval = sorted(eval.items(), key=lambda x: -x[1])
    gradient_list = []
    best_acc = 0

    for idx, (model_path, acc) in enumerate(eval):
        gradient_list.append(model_path)

        if idx == 0:
            best_acc = acc
        else:
            print('eval soup : ', gradient_list)
            soup = uniform_soup(gradient_list)
            model.load_state_dict(soup)
            new_acc = test(model)

            if new_acc > best_acc:
                best_acc = new_acc
            else:
                del gradient_list[-1]

    print('greedy soup ', best_acc, ' : ', gradient_list)
    return uniform_soup(gradient_list)


model_list = ['checkpoint_autoaugment_cifar10/best_acc.pth', 'checkpoint_autoaugment_cifar10/best_loss.pth',
              'checkpoint_autoaugment_SVHN/best_loss.pth', 'checkpoint_autoaugment_imagenet/best.pth', 'checkpoint_autoaugment_imagenet/last.pth']

model_greedy_soup = greedy_soup(model_list)

state = {
    'net': model_greedy_soup
}

torch.save(state, 'soup/greedy_soup.pth')


model_uniform_soup = uniform_soup(model_list)
model.load_state_dict(model_uniform_soup)
new_acc = test(model)

state = {
    'net': model_uniform_soup
}

torch.save(state, 'soup/uniform_soup.pth')
# model.load_state_dict(soup)
# test(model)
