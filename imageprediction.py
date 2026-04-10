import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import numpy as np
import cv2

# Load dataset (built-in CIFAR10)
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()

# Normalize
X_train, X_test = X_train/255.0, X_test/255.0

# Class names
classes = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']

# Build CNN model
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(32,32,3)),
    layers.MaxPooling2D((2,2)),

    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),

    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# Compile
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train
history = model.fit(X_train, y_train, epochs=5, validation_data=(X_test, y_test))

# Evaluate
test_loss, test_acc = model.evaluate(X_test, y_test)
print("Accuracy:", test_acc)

# Graph
plt.plot(history.history['accuracy'], label='train')
plt.plot(history.history['val_accuracy'], label='test')
plt.legend()
plt.title("Model Accuracy")
plt.show()

# Predict on one image
img = X_test[0]
prediction = model.predict(np.expand_dims(img, axis=0))
print("Predicted:", classes[np.argmax(prediction)])

# Show image
plt.imshow(img)
plt.title("Prediction: " + classes[np.argmax(prediction)])
plt.axis('off')
plt.show()
