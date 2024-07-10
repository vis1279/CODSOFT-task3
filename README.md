# CODSOFT-task3
import numpy as np
from tensorflow.keras.datasets import mnist

# Load MNIST dataset
(x_train, ), (, _) = mnist.load_data()

# Normalize pixel values
x_train = x_train.astype('float32') / 255.0

# Reshape data to 4D tensor (batch_size, timesteps, input_dim)
# In this case, timestep will be 28 (each image row)
x_train = np.reshape(x_train, (x_train.shape[0], 28, 28))             
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

model = Sequential()
model.add(LSTM(128, input_shape=(28, 28)))
model.add(Dense(28, activation='sigmoid'))  # Output layer with 28 units (one for each pixel)
model.compile(loss='binary_crossentropy', optimizer='adam')
model.fit(x_train, x_train, batch_size=128, epochs=10)
# Seed image (initial input to generate new sequence)
seed_image = x_train[0]  # Assuming using the first image in the dataset as seed

# Generate new handwritten-like digits
generated_images = []
current_image = seed_image

for _ in range(10):  # Generate 10 images
    # Reshape current image to match model input shape
    current_image = np.reshape(current_image, (1, 28, 28))
    
    # Predict the next image in the sequence
    next_image = model.predict(current_image)
    
    # Append generated image to list
    generated_images.append(next_image.reshape(28, 28))
    
    # Update current image to be the predicted image for the next iteration
    current_image = next_image

# Show generated images (plotting code depends on your preference, e.g., matplotlib)
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 5))
for i in range(10):
    plt.subplot(2, 5, i+1)
    plt.imshow(generated_images[i], cmap='gray')
    plt.axis('off')
plt.tight_layout()
plt.show()
