from PIL import Image
import numpy as np
from os import listdir
from numpy import asarray
import pickle
from sklearn.utils import shuffle

x_train = np.empty((120, 120), dtype=np.uint8)              # Initial element
x_train = np.expand_dims(x_train, axis=0)                   # (1, 120, 120)

# Add all 'people' images
for filename in listdir("./Dataset/person_120x120/"):
    img = Image.open("./Dataset/person_120x120/" + filename).convert('L')
    img = asarray(img)
    x_train = np.append(x_train, [img], axis=0)
    print("\r" + str(x_train.shape), end="", flush=True)

x_train = np.delete(x_train, 0, axis=0)                     # Delete initial element
y_train = np.ones((len(x_train)), dtype=np.uint8)           # Create labels

# Add all 'not_people' images
for filename in listdir("./Dataset/not_person_120x120/"):
    img = Image.open("./Dataset/not_person_120x120/" + filename).convert('L')
    img = asarray(img)
    x_train = np.append(x_train, [img], axis=0)
    print("\r" + str(x_train.shape), end="", flush=True)

y_train = np.append(y_train, np.zeros((len(x_train) - len(y_train))))

x_train, y_train = shuffle(x_train, y_train)                # Shuffle the dataset

print(x_train.shape)
with open("./Dataset/dataset_array.pickle", "wb") as f:
    pickle.dump([x_train, y_train], f)

print("Shape of the dataset: " + str(x_train.shape))            # (25053, 120, 120)
print("Shape of the labels: " + str(y_train.shape))             # (25053, )
print("The data type of the dataset: " + str(x_train.dtype))    # uint8