# TensorFlow 2 quickstart for beginners
# - https://www.tensorflow.org/tutorials/quickstart/beginner
#
# This short introduction uses Keras to:
#
# 1. Build a neural network that classifies images.
# 2. Train this neural network.
# 3. And, finally, evaluate the accuracy of the model.

import tensorflow as tf
from tensorflow.python.keras.layers.core import Dropout

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train/255.0, x_test/255.0  # convert to floats in [0,1]

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),  # Dropout counts for the layer above(?)
    tf.keras.layers.Dense(10)
])

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
# Note: "If you want to provide labels using one-hot representation, please use
#       CategoricalCrossentropy loss"
# Note: Use `from_logits=False` (default) if using probability distribution
#       (e.g. Softmax layer) as output.
# - API reference.

model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5)

model.evaluate(x_test, y_test, verbose=2)
# Note: According to API reference, verbose can only be 0 or 1.
#       * 0: silent
#       * 1: progress bar

# Wrapper for probability model.
# Note: "It is possible to bake this tf.nn.softmax in as the activation function
#       for the last layer of the network. While this can make the model output
#       more directly interpretable, this approach is discouraged as it's
#       impossible to provide an exact and numerically stable loss calculation
#       for all models when using a softmax output."
# - Tutorial page.
prob_model = tf.keras.Sequential([
    model,
    tf.keras.layers.Softmax()
])
