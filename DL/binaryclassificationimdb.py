import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, GlobalAveragePooling1D
from tensorflow.keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt

# Step 2: Load the IMDB Dataset (only keep the top 10,000 words)
vocab_size = 10000
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=vocab_size)

# Step 3: Pad the sequences (make all reviews the same length)
maxlen = 256  # max length of review
x_train = pad_sequences(x_train, maxlen=maxlen)
x_test = pad_sequences(x_test, maxlen=maxlen)

# Step 4: Build the Deep Neural Network
model = Sequential()
model.add(Embedding(input_dim=vocab_size, output_dim=16, input_length=maxlen))  # word vectors
model.add(GlobalAveragePooling1D())  # compress review
model.add(Dense(16, activation='relu'))  # hidden layer
model.add(Dense(1, activation='sigmoid'))  # output layer for binary classification

# Step 5: Compile the Model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Step 6: Train the Model
history = model.fit(x_train, y_train,
                    epochs=10,
                    batch_size=512,
                    validation_split=0.2,
                    verbose=1)

# Step 7: Plot the Accuracy and Loss
plt.plot(history.history['accuracy'], label='train acc')
plt.plot(history.history['val_accuracy'], label='val acc')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Model Accuracy')
plt.show()

# Step 8: Evaluate the Model
loss, acc = model.evaluate(x_test, y_test)
print(f"Test Accuracy: {acc:.4f}")
