import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

fashion_mnist = tf.keras.datasets.fashion_mnist.load_data()
(X_train, y_train), (X_test, y_test) = fashion_mnist

# reshape our data
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)/255
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)/255

model = models.Sequential([
    layers.Conv2D(32,(3,3),activation='relu',input_shape=(28,28,1)),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64,(3,3),activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Flatten(),
    layers.Dense(64,activation='relu'),
    layers.Dense(10,activation='softmax')
])

model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

model.summary()

# train the model
history = model.fit(X_train,y_train,epochs=10,validation_split=0.2,validation_data=(X_test,y_test),verbose=0)

plt.plot(history.history['accuracy'],label ='Training Accuracy')
plt.plot(history.history['val_accuracy'],label ='Validation Accuracy')
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

# model is overfitting

test_loss, test_acc = model.evaluate(X_test, y_test)
print('Test accuracy:', test_acc)

