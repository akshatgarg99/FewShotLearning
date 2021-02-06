import os
import random


def cat_loader(data_loc):
    # read the training folders and create a list of locations
    cat_list = os.listdir(data_loc)
    cat_list.sort(key=int)

    # divide training and validation categories
    # data will be validated on 390 unseen categories
    val_categories = 390
    train_categories = 1796  # chosen so that it is divisible by 32

    # divide the dataset
    val_cat_list = cat_list[:val_categories]
    train_cat_list = cat_list[val_categories:]
    return train_cat_list, val_cat_list

def get_image_loc(cat_list,data_loc):
    image_loc = []
    for i in cat_list:
        image_list = os.listdir(data_loc + i)

        # add more images by randomly selecting images from the class and flipping it
        # to make the dataset balanced
        extra = 50 - len(image_list)
        sample = random.sample(image_list, extra)
        for l in sample:
            # add e (extra to its name)
            image_list.append(l + 'e')
        for j in image_list:
            image_loc.append(data_loc + i + '/' + j)
    return image_loc


def image_loader(data_loc)
    train_cat_list, val_cat_list = cat_loader(data_loc)
    # get the location of the images for the training set
    train_image_loc = get_image_loc(train_cat_list, data_loc)
    val_image_loc = get_image_loc(val_cat_list, data_loc)
    return train_image_loc, val_image_loc