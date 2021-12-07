import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import pickle
import matplotlib.pyplot as plt
from datetime import datetime

with open("./Dataset/dataset_array.pickle", "rb") as dataset_file:
    x_train, y_train = pickle.load(dataset_file)

mask = list(range(22000, len(y_train) - 100))   # leave 100 samples for validation
x_val = x_train[mask]
y_val = y_train[mask]
x_val = np.expand_dims(x_val, axis=3)

mask = list(range(0, 22000))
x_train = x_train[mask]
y_train = y_train[mask]
x_train = np.expand_dims(x_train, axis=3)

x_mean = np.around(np.mean(x_train, axis=0))
x_train = x_train - x_mean
x_val = x_val - x_mean

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(120, 120, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(2, activation='softmax'))

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['accuracy'])

# Variables used for naming the files
batch_size = 128
epochs = 5

history = model.fit(
    x_train,
    y_train,
    batch_size=batch_size,
    epochs=epochs,
    validation_data=(x_val, y_val))

baseline_model_accuracy = model.evaluate(x_val, y_val, verbose=0)

# History for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig("./Training_plots/accuracy_" +
            str(batch_size) + "b-" +
            str(epochs) + "e-" +
            datetime.now().strftime("%d%m%Y%H%M") + ".png")
plt.show()

# History for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig("./Training_plots/loss_" +
            str(batch_size) + "b-" +
            str(epochs) + "e-" +
            datetime.now().strftime("%d%m%Y%H%M") + ".png")
plt.show()

model.save("./Models/person_presence_model_" + datetime.now().strftime("%m%d%H%M") + ".h5")
