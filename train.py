from torchvision import models
import torch.nn as nn
import torch
import shutil
import time
import numpy as np

from Utils.train_val_split import image_loader
from Models.model import Classifier
from Utils.Data import DataSets
from Utils.sampler import Sample_Generator
from Utils.protonet_calculations import proto_net_episode


def pretrained_model(output_vector=512, add_conv_block=False):
    clf = models.resnet18(pretrained=True)
    for param in clf.parameters():
        param.requires_grad = False

    if add_conv_block:
        clf.avgpool = nn.Sequential(
            nn.Conv2d(512, 1024, 3, padding=1),
            nn.BatchNorm2d(1024),
            nn.AdaptiveAvgPool2d(output_size=(1, 1)))
        clf.fc = nn.Linear(1024, output_vector)
    else:
        clf.fc = nn.Linear(512, output_vector)
    return clf


# save the model at checkpoints
def save_ckp(state, is_best, checkpoint_path, best_model_path):
    # save any model
    f_path = checkpoint_path

    torch.save(state, f_path)

    # save only if the model parameters give max accuracy uotil then
    if is_best:
        best_fpath = best_model_path

        shutil.copyfile(f_path, best_fpath)


def load_ckp(checkpoint_fpath, model, optimizer):
    # load saved model

    checkpoint = torch.load(checkpoint_fpath)
    model.load_state_dict(checkpoint['state_dict'])

    optimizer.load_state_dict(checkpoint['optimizer'])
    valid_acc_max = checkpoint['valid_acc_max']

    return model, optimizer, checkpoint['epoch'], valid_acc_max.item()


def train(epochs, data_loc, n_shot=4, q_queries=1, k_way=32, pretarined=True):
    num_train_cat = 1796
    num_val_cat = 390

    data = image_loader(data_loc, num_train_cat, num_val_cat)
    train_image_loc, val_image_loc, train_cat_list, val_cat_list = data
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    if pretarined:
        model = pretrained_model()
    else:
        model = Classifier()
    model = model.to(device)
    print('Reading Files...')

    # create train and validation set
    train_dataset = DataSets(train_image_loc, train_cat_list)
    valid_dataset = DataSets(val_image_loc, val_cat_list)

    train_sampler = Sample_Generator(train_dataset, num_categories=num_train_cat, n=n_shot, q=q_queries, k=k_way)
    val_Sampler = Sample_Generator(valid_dataset, num_categories=num_val_cat, n=n_shot, q=q_queries, k=k_way)

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_sampler=train_sampler, num_workers=16)
    valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_sampler=val_Sampler, num_workers=8)

    loss_fn = torch.nn.CrossEntropyLoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=1e-4)

    # expected accuracy > 80%
    valid_acc_max = 80

    checkpoint_path = "/content/drive/MyDrive/Project_Data/RetailPulse/model_1/b/current_checkpoint.pt"
    best_model_path = "/content/drive/MyDrive/Project_Data/RetailPulse/model_1/b/best_model.pt"

    print('Loading Data Begins...')
    # Record all the losses in an epoch
    losses = []
    mstone = [5, 9, 11, 13]
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimiser, milestones=mstone,
                                                     gamma=0.1, verbose=True)

    for epoch in range(1, epochs + 1):
        print('----------------------------------------------------------------------------------')
        print('======== Starting Epoch Number:', epoch, ' ========')
        start = time.time()
        for batch_index, batch in enumerate(train_dataloader):
            Train = True

            x, _ = batch
            y = torch.arange(0, k_way, 1 / q_queries).long()

            x, y = x.to(device), y.to(device)

            # take one step
            loss, y_pred = proto_net_episode(model, optimiser, loss_fn, x, y, n_shot, k_way, q_queries, Train)
            losses.append(loss.item())
            end = time.time()

            if batch_index % 200 == 0:
                print('Batch Index : %d Loss : %.3f Time : %.3f seconds ' % (batch_index, np.mean(losses), end - start))
                start = time.time()
                correct = 0
                total = 0

                # evaluation
                with torch.no_grad():

                    Train = False

                    for batch_index, batch in enumerate(valid_dataloader):
                        x, _ = batch
                        y = torch.arange(0, k_way, 1 / q_queries).long()
                        x, y = x.to(device), y.to(device)
                        c, t = proto_net_episode(model, optimiser, loss_fn, x, y, n_shot, k_way, q_queries, Train)
                        correct = correct + c
                        total = total + t

                    valid_acc = 100. * correct / total
                    print(' Test Acc : %.3f' % (valid_acc))

                    # save model
                    checkpoint = {
                        'epoch': epoch + 1,
                        'valid_acc_max': valid_acc,
                        'state_dict': model.state_dict(),
                        'optimizer': optimiser.state_dict(),
                    }
                    save_ckp(checkpoint, False, checkpoint_path, best_model_path)

                    if valid_acc >= valid_acc_max:
                        print('Validation accuracy increased')
                        save_ckp(checkpoint, True, checkpoint_path, best_model_path)
                        valid_acc_max = valid_acc
        scheduler.step()
    return model


if __name__ == '__main__':
    model = train(15, '/content/drive/MyDrive/Project_Data/RetailPulse/train/')
    optimiser = torch.optim.Adam(model.parameters(), lr=1e-4)
    ckp_path = "/content/drive/MyDrive/Project_Data/RetailPulse/model_1/b/best_model.pt"
    model, optimiser, start_epoch, valid_acc_max = load_ckp(ckp_path, model, optimiser)
