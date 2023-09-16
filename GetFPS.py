import time
import numpy as np
import cv2
import tensorflow as tf

# Load the model
model_path = "path/to/model"
model = tf.keras.models.load_model(model_path)

# Initialize input image
input_shape = model.input_shape[1:3]
input_image = np.random.randint(low=0, high=256, size=(1, *input_shape, 3)).astype(np.uint8)

# Warm up the model
model.predict(input_image)

# Set up the FPS counter
num_frames = 1000
start_time = time.time()

# Run the inference for num_frames
for i in range(num_frames):
    model.predict(input_image)

# Calculate the FPS
end_time = time.time()
fps = num_frames / (end_time - start_time)

# Print the FPS
print(f"FPS: {fps:.2f}")
