import tensorflow as tf
import matplotlib.pyplot as plt

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

print(train_images.shape)
print(test_images.shape)

# Zeigen Sie die ersten 25 Bilder im Trainingsdatensatz an
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.imshow(train_images[i], cmap='gray')
    plt.title(train_labels[i])
    plt.axis('off')
plt.show()
plt.savefig('mnist.png')

train_images = train_images.reshape((60000, 28 * 28))
train_images.astype('float32') / 255

train_images = train_images.reshape((10000, 28 * 28))
train_images.astype('float32') / 255



train_labels = tf.keras.utils.to_categorical(train_labels)