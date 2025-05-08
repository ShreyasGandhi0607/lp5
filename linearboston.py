import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Load the Boston Housing dataset
(x, y), (x_test, y_test) = tf.keras.datasets.boston_housing.load_data()

# Standardize features
scaler = StandardScaler()
x = scaler.fit_transform(x)
x_test = scaler.transform(x_test)

# Split training data again for validation
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42)

# Build the model
model = tf.keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(x_train.shape[1],)),
    layers.Dense(64, activation='relu'),
    layers.Dense(1)
])

# Compile the model
model.compile(loss='mse', metrics=['mean_absolute_error'])

# Train the model
history = model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=50)

# Evaluate the model
loss, mae = model.evaluate(x_test, y_test)
print("Mean Absolute Error on Test Set:", mae)

# Plot training & validation loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(['Training', 'Validation'])
plt.show()
