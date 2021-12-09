import numpy as np
import tensorflow as tf
import pickle
from os import listdir
from os.path import splitext
from matplotlib import pyplot as plt
import random

# Load the dataset to create the representative dataset
with open("./Dataset/dataset_array.pickle", "rb") as dataset_file:
    x, y = pickle.load(dataset_file)

x = np.expand_dims(x, axis=3)
x_mean = np.around(np.mean(x, axis=0))
x = x - x_mean

# User input: choose model file to test:
files = [file for file in listdir("./Models") if splitext(file)[1] == ".tflite"]
print("Choose one file by number:")
counter = 0
model_file = ""
while model_file not in files:
    for file in files:
        counter += 1
        print(str(counter) + ": " + file)
    model_file = files[int(input()) - 1]
    counter = 0

interpreter = tf.lite.Interpreter(model_path="./Models/" + model_file)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


def invoke_model(x_test):
    input_data = tf.cast(np.array(x_test), tf.int8)
    input_data = np.expand_dims(input_data, axis=0)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    return output_data
"""
correct = 0
for i in range(100):
    result = invoke_model(x[i])
    if (result[0][0] > result[0][1] and y[i] == 0) or ( result[0][1] > result[0][0] and y[i] == 1):
        correct += 1

print(correct)
"""
for i, idx in enumerate(random.sample(range(22000, 25000), 16)):
    result = invoke_model(x[idx])
    plt.subplot(4, 4, i + 1)
    plt.tight_layout(pad=1)
    plt.imshow(x[idx], cmap='gray')
    plt.axis('off')
    if result[0][1] >= result[0][0]:
        title = 'human %d' % result[0][1]
    elif result[0][0] >= result[0][1]:
        title = 'not human: %d' % result[0][0]

    if y[idx] == 1:
        title = title + '\nhuman'
    else:
        title = title + '\nnot human'
    plt.title(title, fontdict={'fontsize': 10})
    print(result[0])

plt.show()