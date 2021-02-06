# calculate the mean and the std for the dataset
# mean and std calculation

import torchvision.transforms as transforms
import torch
import cv2
import os


def process_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = transforms.ToTensor()(image)
    image = transforms.Resize((256, 256))(image)
    return image


class DataSets(torch.utils.data.Dataset):
    def __init__(self, location_list, trans=None):
        self.location_list = location_list

    def __len__(self):
        return len(self.location_list)

    def __getitem__(self, idx):
        loc = self.location_list[idx]
        image = cv2.imread(loc)
        image = process_image(image)

        return image


def get_values(data_loc):
    cat_list = os.listdir('data_loc')
    cat_list.sort(key=int)

    image_loc = []
    for i in cat_list:
        image_list = os.listdir(data_loc + i)
        for j in image_list:
            image_loc.append(data_loc + i + '/' + j)
    dataset = DataSets(image_loc)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=100,
        num_workers=16,
        shuffle=False
    )
    mean = 0.
    std = 0.
    nb_samples = 0.
    for idx, data in enumerate(loader):
        print(idx)
        batch_samples = data.size(0)
        data = data.view(batch_samples, data.size(1), -1)
        mean += data.mean(2).sum(0)
        std += data.std(2).sum(0)
        nb_samples += batch_samples

    mean /= nb_samples
    std /= nb_samples
    return mean, std
