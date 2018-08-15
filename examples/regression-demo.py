'''
use simple equation of linear regression to generate data for train and test
'''
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np

def createFs(n):
  res = []
  r = []
  for i in range(n):
    res.append(np.random.randint(1, 10000) * 0.01)
    x = np.random.randint(1, 10000) * 0.01
    print(x)
    r.append(
      (int(x * 0.2) + 1, int(x * 1.8) + 2)
    )
  return (res, r)

# how many x for equation
XNUM = 10

# f's for x's
fs, xrangeArr = createFs(XNUM)

def createItem():
  '''
  create one random data,
  v = f1 * x1 + f2 * x2 + f3 * x3....
  '''
  data = []
  y = 0
  for i in range(XNUM):
    f = fs[i]
    r = xrangeArr[i]
    x = np.random.randint(r[0], r[1])
    y = y + x * f
    data.append(x)
  return (data, y)

def createData(n):
  '''
  create data array
  '''
  data = []
  labels = []
  for i in range(n):
    d, label = createItem()
    data.append(d)
    labels.append(label)

  return (np.array(data), np.array(labels))

def main():

  print('tensorflow version:', tf.__version__)

  (train_data, train_labels) = createData(6000)
  (test_data, test_labels) = createData(1000)


  print("Training set: {}".format(train_data.shape))  # 404 examples, 13 features
  print("Testing set:  {}".format(test_data.shape))   # 102 examples, 13 features
  print(train_labels[:10])
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

  model.fit(
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
  print('Testing set Mean Abs Error: {}'.format(mae))
  print('Testing loss: {}'.format(loss))


if __name__ == '__main__':
  main()