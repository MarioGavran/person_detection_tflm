import numpy as np
import tensorflow as tf
import pickle
from os import listdir
from os.path import splitext
from matplotlib import pyplot as plt
import random
from datetime import datetime


# User input: choose model file to quantize:
files = [file for file in listdir("./Models") if splitext(file)[1] == ".h5"]
print("Choose one file by number:")
counter = 0
model_file = ""
while model_file not in files:
    for file in files:
        counter += 1
        print(str(counter) + ": " + file)
    model_file = files[int(input()) - 1]
    counter = 0

# Load the model
model = tf.keras.models.load_model("./Models/" + model_file)

# Load the dataset to create the representative dataset
with open("./Dataset/dataset_array.pickle", "rb") as dataset_file:
    x, y = pickle.load(dataset_file)

rnd_idx = random.sample(range(0, len(y)), 200)
x = x[rnd_idx]

x = np.expand_dims(x, axis=3)
x = tf.cast(x, tf.float32)

x_mean = np.around(np.mean(x, axis=0))
x = x - x_mean

#y = np.expand_dims(y, axis=1)


def representative_dataset_gen():
    for input_value in tf.data.Dataset.from_tensor_slices(x).batch(1).take(100):
        yield [input_value]


converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.uint8

tflite_model = converter.convert()

open("./Models/person_presence_quantized_model_" + datetime.now().strftime("%d%m%H%M") + ".tflite", "wb").write(tflite_model)


