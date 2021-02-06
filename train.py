from torchvision import models
import torch.nn as nn
import torch
import shutil


def pretrained_model1(output_vector=512, add_conv_block=False):
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

