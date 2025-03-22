import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from scipy.ndimage import gaussian_filter, convolve

import matplotlib.pyplot as plt
# 1. Load and preprocess the MNIST dataset
def load_and_preprocess_data():
    # load the mnist dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # normalize pixel values to the range [0, 1]
    x_train, x_test = x_train / 255.0, x_test / 255.0

    # add a channel dimension to the images
    x_train = x_train[..., np.newaxis]
    x_test = x_test[..., np.newaxis]

    # convert labels to one-hot encoding
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    # split the training data into training and validation sets (80/20 split)
    split_idx = int(0.8 * len(x_train))
    x_train, x_val = x_train[:split_idx], x_train[split_idx:]
    y_train, y_val = y_train[:split_idx], y_train[split_idx:]

    return x_train, y_train, x_val, y_val, x_test, y_test

# visualize random mnist samples
def visualize_samples(x_data, y_data, num_samples=5):
    # randomly select a few samples to display
    indices = np.random.choice(len(x_data), num_samples, replace=False)
    for i, idx in enumerate(indices):
        plt.subplot(1, num_samples, i + 1)
        plt.imshow(x_data[idx].squeeze(), cmap='gray')
        plt.title(np.argmax(y_data[idx]))
        plt.axis('off')
    plt.show()

# add gaussian noise to images
def add_gaussian_noise(images, sigma=0.3):
    # add gaussian noise with the specified standard deviation
    noisy_images = images + np.random.normal(0, sigma, images.shape)
    return np.clip(noisy_images, 0, 1)  # ensure pixel values remain in [0, 1]

# sharpen images using a convolutional kernel
def sharpen_images(images):
    # define a sharpening kernel
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])

    # apply the kernel trick to each image
    sharpened_images = np.array([convolve(img.squeeze(), kernel, mode='reflect') for img in images])
    return np.clip(sharpened_images[..., np.newaxis], 0, 1)  # add channel dimension and clip values

# 5. Train CNN model
def build_cnn_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(10, activation='softmax')
    ])
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    return model
## this will actualy call the model and train it
def train_and_evaluate(x_train, y_train, x_val, y_val, x_test, y_test, feature_extractor=None):
    if feature_extractor:
        x_train = feature_extractor(x_train)
        x_val = feature_extractor(x_val)
        x_test = feature_extractor(x_test)

    model = build_cnn_model()
    history = model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=10, batch_size=64, verbose=1)
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    return history, test_accuracy

# 6. Plot validation loss
def plot_validation_loss(histories, labels):
    for history, label in zip(histories, labels):
        plt.plot(history.history['val_loss'], label=label)
    plt.title('Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

# Main script
x_train, y_train, x_val, y_val, x_test, y_test = load_and_preprocess_data()
visualize_samples(x_train, y_train)

# Train models with different feature extractors
histories = []
labels = ['Original', 'Gaussian Noise', 'Sharpened']

# Original images
history, acc_original = train_and_evaluate(x_train, y_train, x_val, y_val, x_test, y_test)
histories.append(history)

# Images with Gaussian noise
history, acc_noisy = train_and_evaluate(x_train, y_train, x_val, y_val, x_test, y_test, add_gaussian_noise)
histories.append(history)

# Sharpened images
history, acc_sharpened = train_and_evaluate(x_train, y_train, x_val, y_val, x_test, y_test, sharpen_images)
histories.append(history)

# Plot validation loss
plot_validation_loss(histories, labels)

# Report final accuracies
print(f"Test Accuracy (Original): {acc_original:.4f}")
print(f"Test Accuracy (Gaussian Noise): {acc_noisy:.4f}")
print(f"Test Accuracy (Sharpened): {acc_sharpened:.4f}")