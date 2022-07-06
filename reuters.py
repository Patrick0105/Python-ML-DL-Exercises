# -*- coding: utf-8 -*-
"""reuters.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1qpz2Ygdeh3FIbQlAxuQlxlMD3L-kPzLG
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
reuters = tf.keras.datasets.reuters
(train_text, train_labels), (test_text, test_labels) = reuters.load_data()

train_labels

train_text

reuters_dict = reuters.get_word_index()
reuters_dict.items()

reuters_dict = {key: (value+3) for key, value in reuters_dict.items()}
reuters_dict ['<PAD>'] = 0
reuters_dict['<START>'] = 1
reuters_dict['<UNKNOWN>'] = 2
reuters_dict['<UNUSED>'] = 3

reuters_dict_keys = list(reuters_dict.keys())
reuters_dict_values = list(reuters_dict.values())

example_text = []
for word_index in train_text[0]:
  id = reuters_dict_values.index(word_index)
  word = reuters_dict_keys[id]
  example_text.append(word)
  #example_text.append(reuters_dict_keys[reuters_dict_values.index(word_index)])
print(train_text[0])
print(example_text)
print(train_labels[0])

print(reuters_dict['<PAD>'])
print(reuters_dict['<START>'])
print(reuters_dict['<UNKNOWN>'])
print(reuters_dict['<UNUSED>'])

print(len(train_text[0]))
print(len(train_text[1]))

from tensorflow.keras.preprocessing.sequence import pad_sequences 

train_text = pad_sequences(train_text,value=reuters_dict['<PAD>'], padding='post', maxlen=100)
test_text = pad_sequences(test_text, value=reuters_dict['<PAD>'], padding='post', maxlen=100)
print(len(train_text[0]))
print(len(train_text[1]))

model = tf.keras.Sequential([
  #tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(1280, activation='relu'),
  #tf.keras.layers.Dense(10)
  tf.keras.layers.Dense(46)
])

model = tf.keras.Sequential([
  tf.keras.layers.Embedding(input_dim=len(reuters_dict_keys), output_dim=100),
  tf.keras.layers.GlobalAveragePooling1D(),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(2, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

test_text[0]

test_labels[0]

example_text = []
for word_index in test_text[0]:
  example_text.append(reuters_dict_keys[reuters_dict_values.index(word_index)])

print(' '.join(example_text))