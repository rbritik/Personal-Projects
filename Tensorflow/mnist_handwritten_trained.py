import tensorflow as tf  
from tensorflow import keras
import cv2 as cv
import numpy as np 
import matplotlib.pyplot as plt
mnist_digits = tf.keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist_digits.load_data()
plt.imshow(train_images[0])
#cv.putText(train_images[0], cv.FONT_ST)
#plt.show()
train_images = train_images/255.0
test_images = test_images/255.0
train_images = train_images.reshape(train_images.shape[0],28,28,1)
test_images = test_images.reshape(test_images.shape[0],28,28,1)

# Defining model
model = tf.keras.Sequential([
    keras.layers.Conv2D(64,(3,3),activation = 'relu', input_shape = (28,28,1) ),
    keras.layers.MaxPooling2D((2,2)),
    keras.layers.Conv2D(64,(3,3),activation = 'relu' ),
    keras.layers.Flatten(),
    keras.layers.Dense(256, activation = 'relu'),
    keras.layers.Dense(10, activation = 'softmax')
])

# Compiling the model
model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])

# defining callbacks
class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs = {}):
        if logs.get('loss') < 0.05:
            print("98% accuracy attained")
            self.model.stop_training = True
callbacks = myCallback()

# Trainig the model
model.fit(train_images, train_labels, epochs = 10, callbacks = [callbacks])
print(model.summary())
print("accuracy on test data",model.evaluate(test_images, test_labels))
model.save('mnist.h5')
print("Saving model")