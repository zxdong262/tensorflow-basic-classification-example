# TensorFlow and tf.keras
import tensorflow as tf
from scipy import misc
from tensorflow import keras
from IPython import get_ipython
# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

print('tensorflow version:', tf.__version__)

ipy = get_ipython()
fashion_mnist = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = [
  'T-shirt/top',
  'Trouser',
  'Pullover',
  'Dress',
  'Coat',
  'Sandal',
  'Shirt',
  'Sneaker',
  'Bag',
  'Ankle boot'
]

train_images = train_images / 255.0

test_images = test_images / 255.0

if ipy is not None:
    ipy.run_line_magic('matplotlib', 'inline')

plt.figure(figsize=(10,10))
for i in range(25):
  plt.subplot(5,5,i+1)
  plt.xticks([])
  plt.yticks([])
  plt.grid(False)
  plt.imshow(train_images[i], cmap='binary')
  plt.xlabel(class_names[train_labels[i]])

model = keras.Sequential([
  keras.layers.Flatten(input_shape=(28, 28)),
  keras.layers.Dense(128, activation=tf.nn.relu),
  keras.layers.Dense(10, activation=tf.nn.softmax)
])

model.compile(
  optimizer=tf.train.AdamOptimizer(),
  loss='sparse_categorical_crossentropy',
  metrics=['accuracy']
)

model.fit(train_images, train_labels, epochs=5)

test_loss, test_acc = model.evaluate(test_images, test_labels)

print('Test accuracy:', test_acc)

predictions = model.predict(test_images)

np.argmax(predictions[0])

plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(test_images[i], cmap='binary')
    predicted_label = np.argmax(predictions[i])
    true_label = test_labels[i]
    if predicted_label == true_label:
      color = 'green'
    else:
      color = 'red'
    plt.xlabel("{} ({})".format(
      class_names[predicted_label],
      class_names[true_label]),
      color=color
    )

# Grab an image from the test dataset
# img = test_images[0]

# grab a image to do the test
img = misc.imread('./eg1-sneaker.png', flatten=True)

img = (np.expand_dims(img,0))
predictions1 = model.predict(img)
prediction1 = predictions1[0]

print(
  'predict image as:',
  class_names[
    np.argmax(prediction1)
  ]
)