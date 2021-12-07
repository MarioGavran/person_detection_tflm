import pickle
import random

from PIL import Image
import numpy as np
from os import listdir
from numpy import asarray
from matplotlib import pyplot as plt

with open("./Dataset/dataset_array.pickle", "rb") as dataset_file:
    x_train, y_train = pickle.load(dataset_file)

for i in range(25):
    plt.subplot(5, 5, i + 1)
    idx = random.randint(0,24000)
    plt.imshow(x_train[idx])
    plt.axis('off')
    if y_train[idx] == 1:
        plt.title(1)
    else:
        plt.title(0)

plt.show()
