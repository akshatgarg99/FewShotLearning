import torch
import os
import cv2
import numpy as np

from train import load_ckp
from train import pretrained_model
from Models.model import Classifier
from Utils.Data import process_image
from Utils.protonet_calculations import compute_prototypes, pairwise_distances


pretarined = True
if pretarined:
    model = pretrained_model()
else:
    model = Classifier()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
mod = model.to(device)

optimiser = torch.optim.Adam(model.parameters(), lr=1e-4)
ckp_path = "/content/drive/MyDrive/Project_Data/RetailPulse/model_1/b/current_checkpoint.pt"

model, optimiser, start_epoch, valid_acc_max = load_ckp(ckp_path, mod, optimiser)

print("model = ", model)
print("optimizer = ", optimiser)
print("start_epoch = ", start_epoch)
print("valid_acc_max = ", valid_acc_max)
print("valid_acc_max = {:.6f}".format(valid_acc_max))

# Get the prototypes from the render folder
# get the file locations and the classes
test_cat_list = os.listdir('/content/drive/MyDrive/Project_Data/RetailPulse/test/renders')
test_cat_list.sort(key=int)

# list to add the prototypes for all the categories in the test render set
cat_prototypes = []

# tracker
count = 0

for i in test_cat_list:

    # get lis of all the images in a reder folder
    image_loc_list = os.listdir('/content/drive/MyDrive/Project_Data/RetailPulse/test/renders/' + i)
    image_list = []
    count = count + 1

    print('Procesing class number: ', count)

    for j in image_loc_list:
        # read and process the image
        image = cv2.imread('/content/drive/MyDrive/Project_Data/RetailPulse/test/renders/' + i + '/' + j)
        image = process_image(image)
        # add in the list to create a batch
        image_list.append(image)

    print('Total images available for this class is: ', len(image_loc_list))
    print()

    image_tensor = torch.stack(image_list, dim=0).to(device)

    # get the embedings
    with torch.no_grad():
        model.eval()

        image_embeddings = model(image_tensor)

        prototype = compute_prototypes(image_embeddings, 1, image_tensor.shape[0])

        # add the prototypes for all the categories
        cat_prototypes.append(prototype)

# merge them to create a tensor
cat_prototypes = torch.cat(cat_prototypes, dim=0)

# Use the learned prototypes to get the class for the test images
# get the file locations from the test image folder
test_image_list = os.listdir('/content/drive/MyDrive/Project_Data/RetailPulse/test/images')


def image_id(e):
    return e[:-4]


test_image_list.sort(key=image_id)


# create a dataset to load test images
class test_dataset(torch.utils.data.Dataset):
    def __init__(self, loc_list):
        self.loc_list = loc_list

    def __len__(self):
        return len(self.loc_list)

    def __getitem__(self, idx):
        image = cv2.imread('/content/drive/MyDrive/Project_Data/RetailPulse/test/images/' + self.loc_list[idx])
        image = process_image(image)
        return image


test_set = test_dataset(test_image_list)
test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=35, shuffle=False, num_workers=16)

# stack where results of each branch will be recorded
result_stack = []

for idx, batch in enumerate(test_loader):
    batch = batch.to(device)
    print('processing batch Nuber: ', idx)
    with torch.no_grad():
        queries = model(batch)
    distances = pairwise_distances(queries, cat_prototypes)
    result = torch.argmin(distances, dim=1).cpu().numpy()
    result_stack.append(result)

# create one single list for all the results
result_stack = np.concatenate(result_stack)

# replace the results from indices to category ids
result_stack = np.array([test_cat_list[i] for i in result_stack])

# create a dict with imageid as key and category id as value
result_dict = {}
for i in range(44905):
    result_dict[test_image_list[i]] = result_stack[i]

# convert it to a panda dataframe and save a csv file
