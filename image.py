# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
from PIL import Image
# Helper libraries
import numpy as np

print('tensorflow version:', tf.__version__)

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


# Grab an image from the test dataset
img = test_images[0]

img = (np.expand_dims(img, 0))
predictions1 = model.predict(img)
prediction1 = predictions1[0]
print(
  'predict test image as:',
  class_names[
    np.argmax(prediction1)
  ]
)


img1 = 1 - np.array(Image.open('./eg1-sneaker.png')) / 255.0
img1 = (np.expand_dims(img1, 0))
predictions2 = model.predict(img1)
prediction2 = predictions2[0]
print(
  'predict custom image as:',
  class_names[
    np.argmax(prediction2)
  ]
)