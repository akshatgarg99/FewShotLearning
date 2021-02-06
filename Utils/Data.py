import torch
from Utils.mean_std import get_values
import torchvision.transforms as transforms
import random
import cv2


def process_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # calculate from the give data
    mean, std = get_values()

    # imagenet mean and std
    # mean = np.array([0.4914, 0.4822, 0.4465]).reshape(1,1,-1)
    # std  = np.array([0.2023, 0.1994, 0.2010]).reshape(1,1,-1)
    image = (image - mean) / std
    image = transforms.ToTensor()(image)
    image = transforms.Resize((256, 256))(image)
    return image


class DataSets(torch.utils.data.Dataset):
    def __init__(self, location_list, cat_list, trans=None):
        # list of all the file location in the dataset
        self.location_list = location_list
        # list of all the categories of the images
        self.cat_list = cat_list
        self.trans = trans

    def __len__(self):
        return len(self.location_list)

    def __getitem__(self, idx):
        # get the location
        loc = self.location_list[idx]

        # check if it is the extra image and apply transformations
        if loc[-1] == 'e':
            image = cv2.imread(loc[:-1])
            num = random.random()
            if num < 0.5:
                image = image[::-1]
            else:
                image = image[:, ::-1]

        else:
            image = cv2.imread(loc)
        image = process_image(image)

        if self.trans is not None:
            image = self.trans(image)

        # get the category id
        fol = loc.split('/')
        label = self.cat_list.index(fol[-2])
        return image, label
