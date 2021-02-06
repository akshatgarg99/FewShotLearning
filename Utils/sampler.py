import torch
import math
import random


class Sample_Generator(torch.utils.data.Sampler):
    def __init__(self, dataset, num_categories, n=8, q=2, k=16):
        self.dataset = dataset
        self.num_categories = num_categories
        # n_shot(number of images per class)
        self.n = n
        # queries (number of images to the classified)
        self.q = q
        # number of classes
        self.k = k

    def __len__(self):
        # get total number of episodes
        return math.ceil(self.num_categories / self.k) * math.ceil(50 / self.n)

    def __iter__(self):
        # random shuffle the images
        image_shuffle = [i for i in range(50)]
        random.shuffle(image_shuffle)

        # random shuffle the classes
        category_shuffle = [i for i in range(self.num_categories)]
        random.shuffle(category_shuffle)

        # divide the categories into k sized batches [[....k], [.....k],....., [.....k]]
        category_batches = []
        for i in range(0, self.num_categories, self.k):
            # if i+self.k>self.num_categories:
            # category_batches.append(category_shuffle[i:])
            # break
            if i + self.k < self.num_categories + 1:
                category_batches.append(category_shuffle[i:i + self.k])

        # divide the images into n+q sized batches [[....n+q], [.....n+q],....., [.....n+q]]
        image_batches = []
        for i in range(0, 50, self.n + self.q):
            # if i+self.n+self.q>50:
            # image_batches.append(image_shuffle[i:])
            # break
            if i + self.n + self.q < 50:
                image_batches.append(image_shuffle[i:i + self.n + self.q])

        # generate batches with (n+q)*k image per batch with support images occuring first and then query images
        sequence = []
        for i in image_batches:
            for j in category_batches:
                support_batch = []
                query_batch = []
                for k in j:
                    cat_index = k * 50
                    for l in range(len(i)):
                        if l < self.n:
                            support_batch.append(cat_index + i[l])
                        else:
                            query_batch.append(cat_index + i[l])

                sequence.append(support_batch + query_batch)
        return iter(sequence)
