'''
text classification with weight regularization and dropout
'''
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np

print('tensorflow version:', tf.__version__)

imdb = keras.datasets.imdb

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

print("Training entries: {}, labels: {}".format(len(train_data), len(train_labels)))
NUM_WORDS = 10000
def multi_hot_sequences(sequences, dimension):
    # Create an all-zero matrix of shape (len(sequences), dimension)
    results = np.zeros((len(sequences), dimension))
    for i, word_indices in enumerate(sequences):
        results[i, word_indices] = 1.0  # set specific indices of results[i] to 1s
    return results

train_data = multi_hot_sequences(train_data, dimension=NUM_WORDS)
test_data = multi_hot_sequences(test_data, dimension=NUM_WORDS)

# A dictionary mapping words to an integer index
word_index = imdb.get_word_index()

# The first indices are reserved
word_index = {k:(v+3) for k,v in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2  # unknown
word_index["<UNUSED>"] = 3

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

def decode_review(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])

# train_data = keras.preprocessing.sequence.pad_sequences(
#   train_data,
#   value=word_index["<PAD>"],
#   padding='post',
#   maxlen=256
# )

# test_data = keras.preprocessing.sequence.pad_sequences(
#   test_data,
#   value=word_index["<PAD>"],
#   padding='post',
#   maxlen=256
# )

# print(train_data[0])

# input shape is the vocabulary count used for the movie reviews (10,000 words)
vocab_size = 10000
# model = keras.Sequential()
model = keras.Sequential([
    keras.layers.Dense(
      16,
      kernel_regularizer=keras.regularizers.l2(0.001),
      activation=tf.nn.relu, input_shape=(10000,)
    ),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(
      16,
      kernel_regularizer=keras.regularizers.l2(0.001),
      activation=tf.nn.relu
    ),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(1, activation=tf.nn.sigmoid)
])
# model = keras.Sequential([
#   keras.layers.Embedding(vocab_size, 16),
#   keras.layers.GlobalAveragePooling1D(),
#   keras.layers.Dense(16, activation=tf.nn.relu),
#   keras.layers.Dense(1, activation=tf.nn.sigmoid)
# ])

model.summary()

model.compile(
  optimizer='adam',
  loss='binary_crossentropy',
  metrics=['accuracy', 'binary_crossentropy']
)

# x_val = train_data[:10000]
# partial_x_train = train_data[10000:]

# y_val = train_labels[:10000]
# partial_y_train = train_labels[10000:]

history = model.fit(
  train_data,
  train_labels,
  epochs=20,
  batch_size=512,
  validation_data=(test_data, test_labels),
  verbose=1
)


# just copy a negative from imdb
negative = '''<START> This is one of those awful films consisting of too many famous people acting out a plot that has no backbone in the vain hope that their collective fame will patch the holes in the story. I wouldn't wipe my ass with this script.'''

# just copy a positive from imdb
positive = '''<START> I have never seen such an amazing film since I saw The Shawshank Redemption. Shawshank encompasses friendships, hardships, hopes, and dreams. And what is so great about the movie is that it moves you, it gives you hope. Even though the circumstances between the characters and the viewers are quite different, you don't feel that far removed from what the characters are going through.'''

def mapper(x):
  un = word_index.get('<UNK>')
  r = word_index.get(x, un)
  if r > 10000:
    return un
  return r

negativearr = list(map(mapper, negative.split(' ')))
negativearr = multi_hot_sequences([negativearr], NUM_WORDS)

positivearr = list(map(mapper, positive.split(' ')))
positivearr = multi_hot_sequences([positivearr], NUM_WORDS)

print('negative review', negative)


print('positive review', positive)

negativepre = model.predict(
  negativearr
)
positivepre = model.predict(
  positivearr
)
print('predict negative review result:', negativepre)
print('predict positive review result:', positivepre)

print('predict negative review as:', np.average(negativepre))
print('predict positive review as:', np.average(positivepre))
