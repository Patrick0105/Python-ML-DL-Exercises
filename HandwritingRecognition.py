from typing import NoReturn
import tensorflow as tf
mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0


import matplotlib.pyplot as plt
#fig = plt.figure
plt.imshow(x_train[0], cmap='gray')
plt.show()

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28,28)),
    tf.keras.layers.Dense(128, activation = 'relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10)
])
model.summary()

model.compile(
    optimizer = 'adam',
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True),
    metrics = ['accuracy']
)
epoch_num = 10
history = model.fit(x_train, y_train, validation_data = (x_test, y_test), epochs = epoch_num)

epochs = range(1, epoch_num+1) #1,2,3...,10
loss = history.history['loss']
val_loss = history.history['val_loss']

plt.plot(epochs, loss, 'ko', label = 'training loss')
plt.plot(epochs, val_loss, 'ro', label = 'validation loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend()
plt.show()

print(model.predict(x_test[0:3]))
