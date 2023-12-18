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

transform_train = transforms.Compose([
    # transforms.RandomPerspective(distortion_scale=0.2),
    # transforms.RandomApply([
    #             transforms.GaussianBlur((5,9),(0.1,0.3))
    #         ], p=0.5),
    # transforms.RandomRotation(degrees=10),
    transforms.AutoAugment(transforms.AutoAugmentPolicy.SVHN),
    transforms.Resize((448, 448)),
    # transforms.RandomResizedCrop(448, scale=(0.8, 1.2), ratio=(3.0 / 4.0, 4.0 / 3.0)),
    # transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

transform_test = transforms.Compose([
    transforms.Resize((448, 448)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

data_path = '../coco_data/dent/20231208/'
damage = 'dent'

trainset = torchvision.datasets.ImageFolder(
    root=data_path+'train', transform=transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=20, shuffle=True, num_workers=2)

testset = torchvision.datasets.ImageFolder(
    root=data_path+'val', transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=20, shuffle=False, num_workers=2)

model = timm.create_model(
    'eva02_base_patch14_448.mim_in22k_ft_in22k', pretrained=True, num_classes=2)
# print([n for n, _ in model.named_children()])
# for param in model.parameters():
#     param.requires_grad = False

# model.head.weight.requires_grad = True
# model.head.bias.requires_grad = True

# model = timm.create_model('resnest200e', pretrained=True, num_classes=2)

checkpoint_path = 'checkpoint_autoaugment_SVHN'
if not os.path.isdir(checkpoint_path):
    os.mkdir(checkpoint_path)


model.eval()

device = 'cuda:2'
model = model.to(device)
# checkpoint = torch.load('./checkpoint/ckpt_crack.pth')
# model.load_state_dict(checkpoint['net'])

best_acc = 0
best_loss = {'loss': 1, 'epoch': 0}
criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(model.parameters(), lr=0.000002,
#                       momentum=0.9, weight_decay=5e-4)

# optimizer = optim.AdamW(model.parameters(), lr=1e-6, eps=1e-5)

optimizer = optim.SGD(model.parameters(), lr=0.00001,
                      momentum=0.9, weight_decay=5e-4)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=150, eta_min=0.0000008)

# Training


def train(epoch):
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


def test(epoch):
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
        'val_loss':val_loss,
        'epoch': epoch,
    }

    if acc > best_acc:
        torch.save(state, checkpoint_path+'/best_acc.pth')
        best_acc = acc

    if val_loss < best_loss['loss']:
        # print('Saving ...')
        # state = {
        #     'net': model.state_dict(),
        #     'acc': acc,
        #     'epoch': epoch,
        # }

        torch.save(state, checkpoint_path+'/best_loss.pth')
        best_loss['loss'] = val_loss
        best_loss['epoch'] = epoch
    # else:
        # print('Saving..')
    # state = {
    #     'net': model.state_dict(),
    #     'acc': acc,
    #     'epoch': epoch,
    # }
    torch.save(state, checkpoint_path+'/last.pth')


start_epoch = 0
for epoch in range(start_epoch, start_epoch+200):
    print('best loss : ',best_loss)
    train(epoch)
    test(epoch)
    if epoch - best_loss['epoch'] == 4:
        print('Early stopping .... at epoch: ', best_loss['epoch'], ' loss: ',best_loss['loss'])
        break
    
    scheduler.step()

# def transform(example_batch):
#     # Take a list of PIL images and turn them to pixel values
#     inputs = processor([x for x in example_batch['image']], return_tensors='pt')

#     # Don't forget to include the labels!
#     inputs['labels'] = example_batch['label']
#     return inputs

# ds = load_dataset('/data1/dat/ScaledYOLOv4/dataset')
# prepared_ds = ds.with_transform(transform)

# print(ds['train'])
# labels = ds['train'].features['label'].names


# model_name_or_path = 'google/vit-base-patch16-224-in21k'
# processor = ViTImageProcessor.from_pretrained(model_name_or_path)


# model = ViTForImageClassification.from_pretrained(
#     model_name_or_path,
#     num_labels=len(labels),
#     id2label={str(i): c for i, c in enumerate(labels)},
#     label2id={c: str(i) for i, c in enumerate(labels)}
# )


# training_args = TrainingArguments(
#   output_dir="./vit-base-beans",
#   per_device_train_batch_size=1,
#   evaluation_strategy="steps",
#   num_train_epochs=4,
#   fp16=True,
#   save_steps=3000,
#   eval_steps=3000,
#   logging_steps=10,
#   learning_rate=2e-4,
#   save_total_limit=2,
#   remove_unused_columns=False,
#   push_to_hub=False,
#   report_to='tensorboard',
#   load_best_model_at_end=True,
# )


# def collate_fn(batch):
#     return {
#         'pixel_values': torch.stack([x['pixel_values'] for x in batch]),
#         'labels': torch.tensor([x['labels'] for x in batch])
#     }


# metric = load_metric("accuracy")
# def compute_metrics(p):
#     return metric.compute(predictions=np.argmax(p.predictions, axis=1), references=p.label_ids)


# trainer = Trainer(
#     model=model,
#     args=training_args,
#     data_collator=collate_fn,
#     compute_metrics=compute_metrics,
#     train_dataset=prepared_ds["train"],
#     eval_dataset=prepared_ds["validation"],
#     tokenizer=processor,
# )

# train_results = trainer.train()
# trainer.save_model()
# trainer.log_metrics("train", train_results.metrics)
# trainer.save_metrics("train", train_results.metrics)
# trainer.save_state()
