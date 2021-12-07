import tensorflow_model_optimization as tfmot
import tensorflow as tf
import pickle
from os import listdir
from os.path import splitext
import numpy as np
from tensorflow_model_optimization.python.core.sparsity.keras.prune import prune_low_magnitude
import tempfile
from tensorflow_model_optimization.sparsity.keras import UpdatePruningStep
from tensorflow_model_optimization.sparsity.keras import PruningSummaries

# Load the dataset
with open("./Dataset/dataset_array.pickle", "rb") as dataset_file:
    x_train, y_train = pickle.load(dataset_file)


x_train = np.expand_dims(x_train, axis=3)
x_mean = np.around(np.mean(x_train, axis=0))
x_train = x_train - x_mean

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

#prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude

batch_size = 128
epochs = 2
validation_split = 0.1 
num_images = x_train.shape[0] * (1 - validation_split)
end_step = np.ceil(num_images / batch_size).astype(np.int32) * epochs

# Define model for pruning.
pruning_params = {'pruning_schedule': 
                  tfmot.sparsity.keras.PolynomialDecay(
                      initial_sparsity=0.50,
                      final_sparsity=0.80,
                      begin_step=0,
                      end_step=end_step)}

model_for_pruning = prune_low_magnitude(model, **pruning_params)

# `prune_low_magnitude` requires a recompile.
model_for_pruning.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'])

model_for_pruning.summary()

logdir = tempfile.mkdtemp()

callbacks = [
  tfmot.sparsity.keras.UpdatePruningStep(),
  tfmot.sparsity.keras.PruningSummaries(log_dir=logdir),
]

model_for_pruning.fit(
    x_train, y_train,              
    batch_size=batch_size, epochs=epochs, validation_split=validation_split,
    callbacks=callbacks)

model_for_export = tfmot.sparsity.keras.strip_pruning(model_for_pruning)
tf.keras.models.save_model(model_for_export, "./Models/pruned_" + model_file, include_optimizer=False)
