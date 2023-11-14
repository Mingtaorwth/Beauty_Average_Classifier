import torch
from dataclass import MyData
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms
import os
import torchvision
import sys
import torch.nn as nn
from PIL import Image

batch_size = 4
root_dir_train = "archive/train"
root_dir_test = "archive/test"

average = "average"
beautiful = "beautiful"

labels_mark = {"average": 0, "beautiful": 1}


class DataOfCNN(MyData):
    def __init__(self, root_dir, label_dir, train):
        super(DataOfCNN, self).__init__(root_dir, label_dir)
        self.train = train

    def __getitem__(self, idx):
        transform_tensor = transforms.Compose([transforms.ToTensor(),
                                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        image_name = self.img_path[idx]
        img_item_path = os.path.join(self.root_dir, self.label_dir, image_name)
        img = transform_tensor(Image.open(img_item_path))
        label = labels_mark[self.label_dir]
        return img, label


# training set
average_data_train = DataOfCNN(root_dir_train, average, True)
beautiful_data_train = DataOfCNN(root_dir_train, beautiful, True)

# test set
average_data_test = DataOfCNN(root_dir_test, average, True)
beautiful_data_test = DataOfCNN(root_dir_test, beautiful, True)


training_set = average_data_train + beautiful_data_train
test_set = average_data_test + beautiful_data_test

train_data_loader = DataLoader(training_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True)


f, axarr = plt.subplots(2, 2, figsize=(2, 2))

# Load a batch of images into memory
images, labels = next(iter(test_loader))

for i, ax in enumerate(axarr.flat):
    images[i] = images[i] / 2 + 0.5  # de-normalize
    ax.imshow(np.transpose(images[i], (1, 2, 0)))
    ax.axis("off")

im_grid = torchvision.utils.make_grid(images)
# writer.add_image("lady_images", im_grid)
# writer.close()
# sys.exit()
# plt.suptitle("Batch")
# plt.show()

