from __future__ import print_function, division

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.optim import lr_scheduler
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode
from torch.optim.lr_scheduler import CosineAnnealingLR
import model_
from utils import get_yaml_value, save_network, parameter, create_dir
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from pytorch_metric_learning import losses, miners
from tqdm import tqdm

if torch.cuda.is_available():
    device = torch.device("cuda:0")
cudnn.benchmark = True


def one_LPN_output(outputs, labels, criterion, block):
    sm = nn.Softmax(dim=1)
    num_part = block
    score = 0
    loss = 0
    avg_output = 0
    for i in range(num_part):
        part = outputs[i]
        avg_output += part
        loss += criterion(part, labels)
    avg_output /= num_part
    score = sm(avg_output)
    _, preds = torch.max(score.data, 1)
    return preds, loss


def train():
    num_epochs = get_yaml_value("num_epochs")
    drop_rate = get_yaml_value("drop_rate")
    lr = get_yaml_value("lr")
    weight_decay = get_yaml_value("weight_decay")
    model_name = get_yaml_value("model")
    data_dir = get_yaml_value("dataset_path")
    image_size = get_yaml_value("image_size")
    batchsize = get_yaml_value("batch_size")
    weight_save_path = get_yaml_value("weight_save_path")
    fp16 = get_yaml_value("fp16")
    share_weight = get_yaml_value("share_weight")
    LPN = get_yaml_value("LPN")
    block = get_yaml_value("block")

    transform_train_list = [
        transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC),
        transforms.Pad(10, padding_mode='edge'),
        transforms.RandomCrop((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]

    transform_satellite_list = [
        transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC),
        transforms.Pad(10, padding_mode='edge'),
        transforms.RandomAffine(90),
        transforms.RandomCrop((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]

    data_transforms = {
        'train': transforms.Compose(transform_train_list),
        'satellite': transforms.Compose(transform_satellite_list)
    }

    image_datasets = {
        'satellite': datasets.ImageFolder(os.path.join(data_dir, 'satellite'), data_transforms['satellite']),
        'drone': datasets.ImageFolder(os.path.join(data_dir, 'drone'), data_transforms['train'])
    }

    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batchsize,
                                                  shuffle=True, num_workers=32, pin_memory=False)
                   for x in ['satellite', 'drone']}

    dataset_sizes = {x: len(image_datasets[x]) for x in ['satellite', 'drone']}
    print(dataset_sizes)

    class_names = image_datasets['satellite'].classes

    if LPN:
        model = model_.model_dict[model_name](len(class_names), drop_rate, share_weight=share_weight, block=block)
    else:
        model = model_.model_dict[model_name](len(class_names), drop_rate, share_weight=share_weight)

    model = model.cuda()
    criterion_func = losses.TripletMarginLoss(margin=0.3)
    miner = miners.MultiSimilarityMiner()

    if LPN:
        ignored_params = []
        for i in range(block):
            cls_name = 'classifier' + str(i)
            c = getattr(model, cls_name)
            ignored_params += list(map(id, c.parameters()))
        base_params = filter(lambda p: id(p) not in ignored_params, model.parameters())

        optim_params = [{'params': base_params, 'lr': 0.1 * lr}]
        for i in range(block):
            cls_name = 'classifier' + str(i)
            c = getattr(model, cls_name)
            optim_params.append({'params': c.parameters(), 'lr': lr})

        optimizer = optim.RAdam(optim_params, weight_decay=weight_decay)
    else:
        ignored_params = list(map(id, model.classifier.parameters()))
        base_params = filter(lambda p: id(p) not in ignored_params, model.parameters())

        optimizer = optim.RAdam([
            {'params': base_params, 'lr': 0.1 * lr},
            {'params': model.classifier.parameters(), 'lr': lr}
        ], weight_decay=weight_decay)

    if fp16:
        from apex import amp
        model, optimizer = amp.initialize(model, optimizer, opt_level="O2")

    criterion = nn.CrossEntropyLoss()
    scheduler = lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.2)

    print("Dataloader Preprocessing Finished...")

    MAX_LOSS = 0.5
    print("Training Start >>>>>>>>")
    weight_save_name = time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())
    dir_model_name = model_name + "_" + str(1652) + "_" + weight_save_name
    save_path = os.path.join(weight_save_path, dir_model_name)
    create_dir(save_path)
    parameter("name", dir_model_name)
    warm_epoch = 5
    warm_up = 0.1  # We start from the 0.1*lrRate
    warm_iteration = round(dataset_sizes['satellite'] / batchsize) * warm_epoch  # first 5 epoch

    for epoch in range(num_epochs):
        since = time.time()
        running_loss = 0.0
        running_corrects1 = 0.0
        running_corrects2 = 0.0
        total1 = 0.0
        total2 = 0.0
        model.train(True)

        for data1, data2 in zip(dataloaders["satellite"], dataloaders["drone"]):

            input1, label1 = data1
            input2, label2 = data2

            input1 = input1.to(device)
            input2 = input2.to(device)
            label1 = label1.to(device)
            label2 = label2.to(device)

            optimizer.zero_grad()

            output1, output2, _, _ = model(input1, input2)

            total1 += label1.size(0)
            total2 += label2.size(0)

            if LPN:
                preds1, loss1 = one_LPN_output(output1, label1, criterion, block=block)
                preds2, loss2 = one_LPN_output(output2, label2, criterion, block=block)
            else:
                loss1 = criterion(output1, label1)
                loss2 = criterion(output2, label2)

                _, preds1 = torch.max(output1.data, 1)
                _, preds2 = torch.max(output2.data, 1)

            loss = loss1 + loss2

            if epoch < warm_epoch:
                warm_up = min(1.0, warm_up + 0.9 / warm_iteration)
                loss *= warm_up
            if fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_corrects1 += preds1.eq(label1.data).sum()
            running_corrects2 += preds2.eq(label2.data).sum()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        scheduler.step()
        epoch_loss = running_loss / len(class_names)
        satellite_acc = running_corrects1 / total1
        drone_acc = running_corrects2 / total2
        time_elapsed = time.time() - since

        with open(os.path.join(save_path, "log.txt"), "a") as f:
            f.write('[Epoch {}/{}] {} | Loss: {:.4f} | Drone_Acc: {:.2f}% | Satellite_Acc: {:.2f}% | Time: {:.2f}s\n' \
                    .format(epoch + 1, num_epochs, "Train", epoch_loss, drone_acc * 100, satellite_acc * 100,
                            time_elapsed))

        print('[Epoch {}/{}] {} | Loss: {:.4f} | Drone_Acc: {:.2f}% | Satellite_Acc: {:.2f}% | Time: {:.2f}s' \
              .format(epoch + 1, num_epochs, "Train", epoch_loss, drone_acc * 100, satellite_acc * 100, time_elapsed))

        if drone_acc > 0.50 and satellite_acc > 0.50:
            if epoch_loss < MAX_LOSS and epoch > (num_epochs - 10):
                MAX_LOSS = epoch_loss
                save_network(model, dir_model_name, epoch + 1)
                print(model_name + " Epoch: " + str(epoch + 1) + " has saved with loss: " + str(epoch_loss))


if __name__ == '__main__':
    LPN = get_yaml_value("LPN")
    train()
    os.system("/usr/bin/shutdown")
