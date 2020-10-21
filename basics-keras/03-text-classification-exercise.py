# Basic text classification - Exercise
# - https://www.tensorflow.org/tutorials/keras/text_classification
#
# [...]
# At the end of the notebook, there is an exercise for you to try,
# in which you'll train a multiclass classifier to predict the tag for a
# programming question on Stack Overflow.

import matplotlib.pyplot as plt
import os
import re
import string
import tensorflow as tf

from tensorflow.keras import layers
from tensorflow.keras import losses
from tensorflow.keras import preprocessing
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization


# Get dataset
url = "http://storage.googleapis.com/download.tensorflow.org/data/stack_overflow_16k.tar.gz"

dataset = tf.keras.utils.get_file("stack_overflow_16k.tar.gz", url,
                                  untar=True, cache_dir='.',
                                  cache_subdir='stack-overflow')

dataset_dir = os.path.join(os.path.dirname(dataset), 'stack-overflow')
train_dir = os.path.join(dataset_dir, 'train')


# Load training data.

batch_size = 32
seed = 42

raw_train_ds = tf.keras.preprocessing.text_dataset_from_directory(
    'stack-overflow/train',
    batch_size=batch_size,
    validation_split=0.2,
    subset='training',
    seed=seed)

raw_val_ds = tf.keras.preprocessing.text_dataset_from_directory(
    'stack-overflow/train',
    batch_size=batch_size,
    validation_split=0.2,
    subset='validation',  # Note the difference with raw_train_ds definition.
    seed=seed)

raw_test_ds = tf.keras.preprocessing.text_dataset_from_directory(
    'stack-overflow/test',
    batch_size=batch_size)

for i in range(4):
    print(f"Label {i} corresponds to {raw_train_ds.class_names[i]}")


# Standardize, tokenize, and vectorize the data

# max_features = 10000 # Official solution uses 5000
max_features = 5000
# sequence_length = 250 # Official solution uses 500
sequence_length = 500

vectorize_layer = TextVectorization(
    max_tokens=max_features,
    output_mode='int',
    output_sequence_length=sequence_length)

# Make a text-only dataset (without labels), then call adapt
train_text = raw_train_ds.map(lambda x, y: x)
vectorize_layer.adapt(train_text)


def vectorize_text(text, label):
    text = tf.expand_dims(text, -1)
    return vectorize_layer(text), label


train_ds = raw_train_ds.map(vectorize_text)
val_ds = raw_val_ds.map(vectorize_text)
test_ds = raw_test_ds.map(vectorize_text)


# Note: .cache and .prefetch are called to ensure that IO does not become
# a blocking factor for the training step.
AUTOTUNE = tf.data.experimental.AUTOTUNE

train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)


# The model
# embedding_dim = 16 # Official solution uses 128
embedding_dim = 128

model = tf.keras.Sequential([
    layers.Embedding(max_features + 1, embedding_dim),
    layers.Dropout(0.2),
    layers.GlobalAveragePooling1D(),  # Average input to single 16 dim encoding
    layers.Dropout(0.2),
    layers.Dense(4)  # Single output node; no one-hot output.
])

model.compile(loss=losses.SparseCategoricalCrossentropy(from_logits=True),
              optimizer='adam',
              metrics=['accuracy'])


epochs = 10
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs)

loss, accuracy = model.evaluate(test_ds)

print("Loss: ", loss)
print("Accuracy: ", accuracy)


# Plot training.
history_dict = history.history
acc = history_dict['accuracy']
val_acc = history_dict['val_accuracy']
loss = history_dict['loss']
val_loss = history_dict['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, loss, 'r', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

plt.plot(epochs, acc, 'r', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')

plt.show()


# Export the model.
export_model = tf.keras.Sequential([
    vectorize_layer,  # Add the vectorisation layer before the input.
    model,
    layers.Activation('sigmoid')
])

export_model.compile(
    loss=losses.SparseCategoricalCrossentropy(from_logits=False),
    optimizer="adam",
    metrics=['accuracy']
)

# Test it with `raw_test_ds`, which yields raw strings
loss, accuracy = export_model.evaluate(raw_test_ds)
print(f"Test accuracy on raw data via the exported model: {accuracy}")

# Accuracy with hyperparameters as in the IMDB classification: ~0.74
# Accuracy with hyperparameters as in the official solution: ~0.76
