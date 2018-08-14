'''
from https://www.tensorflow.org/tutorials/keras/basic_regression
'''
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np

print(tf.__version__)

boston_housing = keras.datasets.boston_housing

(train_data, train_labels), (test_data, test_labels) = boston_housing.load_data()


print("Training set: {}".format(train_data.shape))  # 404 examples, 13 features
print("Testing set:  {}".format(test_data.shape))   # 102 examples, 13 features

print(train_labels[0:10])

mean = train_data.mean(axis=0)
std = train_data.std(axis=0)
train_data = (train_data - mean) / std
test_data = (test_data - mean) / std
print('mean:', mean)
print('std:', std)

def build_model():

  model = keras.Sequential([
    keras.layers.Dense(
      64,
      activation=tf.nn.relu,
      input_shape=(train_data.shape[1],)
    ),
    keras.layers.Dense(64, activation=tf.nn.relu),
    keras.layers.Dense(1)
  ])

  optimizer = tf.train.RMSPropOptimizer(0.001)

  model.compile(
    loss='mse',
    optimizer=optimizer,
    metrics=['mae']
  )
  return model

model = build_model()
model.summary()

# Display training progress by printing a single dot for each completed epoch.
class PrintDot(keras.callbacks.Callback):
  def on_epoch_end(self,epoch,logs):
    if epoch % 100 == 0: print('')
    print('.', end='')

EPOCHS = 500

# The patience parameter is the amount of epochs to check for improvement.
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=20)

history = model.fit(
  train_data,
  train_labels,
  shuffle=True,
  epochs=EPOCHS,
  validation_split=0.3,
  verbose=0,
  callbacks=[early_stop, PrintDot()]
)

[loss, mae] = model.evaluate(test_data, test_labels, verbose=0)

print('')
print('Testing set Mean Abs Error: ${:7.2f}'.format(mae * 1000))
print('Testing loss: {}'.format(loss))
print('predict test data:')

test_predictions = model.predict(test_data).flatten()

print(test_predictions)
