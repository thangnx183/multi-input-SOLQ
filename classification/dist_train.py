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
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import timm


def ddp_setup(rank, world_size):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

    print('rank, world size : ', rank, world_size)


transform_train = transforms.Compose([

    transforms.AutoAugment(transforms.AutoAugmentPolicy.IMAGENET),
    # transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10),
    # transforms.AutoAugment(transforms.AutoAugmentPolicy.SVHN),
    transforms.Resize((448, 448)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

transform_test = transforms.Compose([
    transforms.Resize((448, 448)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

batch_size = 44

data_path = '../coco_data/dent/20231208/'
damage = 'dent'

checkpoint_path = 'checkpoint_autoaugment_IMAGENET'
if not os.path.isdir(checkpoint_path):
    os.mkdir(checkpoint_path)


best_acc = 0
best_loss = {'loss': 1, 'epoch': 0}

def train_1_epoch(epoch, model, optimizer, criterion, trainloader, device):
    if device in {-1, 0}:
        print('\nEpoch: %d' % epoch)
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):

        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))


def test(epoch, model, criterion, testloader, device):
    global best_acc
    global best_loss
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            val_loss = test_loss/(batch_idx+1)

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    print('Saving..')

    acc = 100.*correct/total
    state = {
        'net': model.state_dict(),
        'acc': acc,
        'val_loss': val_loss,
        'epoch': epoch,
    }

    if acc > best_acc:
        torch.save(state, checkpoint_path+'/best_acc.pth')
        best_acc = acc

    if val_loss < best_loss['loss']:
        torch.save(state, checkpoint_path+'/best_loss.pth')
        best_loss['loss'] = val_loss
        best_loss['epoch'] = epoch

    torch.save(state, checkpoint_path+'/last.pth')


def main(rank: int, world_size: int):
    ddp_setup(rank, world_size)

    trainset = torchvision.datasets.ImageFolder(
        root=data_path+'train', transform=transform_train)

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size//world_size, shuffle=False, num_workers=2, sampler=DistributedSampler(trainset), pin_memory=True)

    if rank in {-1, 0}:
        testset = torchvision.datasets.ImageFolder(
            root=data_path+'val', transform=transform_test)
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=batch_size//world_size, shuffle=False, num_workers=2)

    model = timm.create_model(
        'eva02_base_patch14_448.mim_in22k_ft_in22k', pretrained=True, num_classes=2)
    model.to(rank)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.0001,
                          momentum=0.9, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=150, eta_min=0.0000008)

    model = DDP(model, device_ids=[rank])

    start_epoch = 0
    for epoch in range(start_epoch, start_epoch+200):
        if rank in {-1, 0}:
            global best_loss
            global best_acc
            print('best loss : ', best_loss, best_acc)
        train_1_epoch(epoch, model, optimizer, criterion, trainloader, rank)

        if rank in {-1, 0}:
            test(epoch, model.module, criterion, testloader, rank)

            if epoch - best_loss['epoch'] == 4:
                print('Early stopping .... at epoch: ',
                      best_loss['epoch'], ' loss: ', best_loss['loss'])
                break

        scheduler.step()

    destroy_process_group()


if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    mp.spawn(main, args=[world_size], nprocs=world_size)
