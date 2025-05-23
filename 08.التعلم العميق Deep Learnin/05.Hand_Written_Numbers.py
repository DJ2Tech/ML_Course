import tensorflow.keras as tk
import matplotlib.pyplot as plt
import numpy as np


mnist = tk.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()



train_images = train_images / 255.0   # /255  as the Numbers for Colrs from 0-255   , then reduce the range 0,1
test_images = test_images / 255.0

model = tk.Sequential([
    tk.layers.Flatten(input_shape=(28, 28)), # Shap 28,28 we checked before 
    tk.layers.Dense(units=512, activation='relu'),
    tk.layers.Dense(units=256, activation='relu'),
    tk.layers.Dense(units=128, activation='relu'),
    tk.layers.Dense(units=10, activation='softmax') # Output SoftMax 
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


model.fit(train_images, train_labels, epochs=10)

test_loss, test_acc = model.evaluate(test_images, test_labels)
print('Test accuracy:', test_acc)


predictions = model.predict(test_images)

plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(test_images[i], cmap=plt.cm.binary)
    
    predicted_label = np.argmax(predictions[i])
    
    true_label = test_labels[i]
    if predicted_label == true_label:
        color = 'green'
    else:
        color = 'red'
    plt.xlabel("{} ({})".format(predicted_label, true_label), color=color)
plt.show()
