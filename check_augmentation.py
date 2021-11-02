import tensorflow as tf
from tensorflow.keras import layers, Sequential
from tensorflow.keras.applications import resnet_v2
from tensorflow.keras.layers.experimental import preprocessing
import matplotlib.pyplot as plt
import numpy as np

IMAGE_SIZE = 244
image_filename = 'indoor_outdoor_images/0-_2hRjVpJtdY.jpg'

resize_and_rescale = tf.keras.Sequential([
  layers.experimental.preprocessing.Resizing(IMAGE_SIZE, IMAGE_SIZE),
  layers.experimental.preprocessing.Rescaling(1./255)
])

data_augmentation = Sequential([
    # preprocessing.RandomFlip("horizontal", input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3), seed=42),
    preprocessing.RandomRotation(factor=(0, 0), seed=42),
    preprocessing.RandomZoom(height_factor=(0, 0), width_factor=(0, 0), seed=42),
    preprocessing.RandomTranslation(height_factor=(0, 0), width_factor=(0, 0), seed=42),
])

# Load image and print after resizing, scaling
print("Print original image")
img = tf.io.read_file(image_filename)
original_img = tf.image.decode_jpeg(img, channels=3)
img_plot = plt.imshow(original_img)
plt.axis("off")
plt.show()

print("Print processed image - resized, rescaled")
processed_image = resize_and_rescale(original_img)
img_plot = plt.imshow(processed_image)
plt.axis("off")
plt.show()

# Print image when scaled using Resnet preprocess_input function
print("Print resnet processed image")
image = tf.image.resize(original_img, [IMAGE_SIZE, IMAGE_SIZE])
# recast_image = tf.image.convert_image_dtype(image, tf.float32)
resnet_image = resnet_v2.preprocess_input(image)
img_plot = plt.imshow(resnet_image)
plt.axis("off")
plt.show()

print("Print augmented image")
img = tf.expand_dims(original_img, 0)
img = tf.image.resize(img, [IMAGE_SIZE, IMAGE_SIZE])
augmented_image = data_augmentation(img)
preprocessed_augmented_image = resnet_v2.preprocess_input(augmented_image)
plt.imshow(preprocessed_augmented_image[0])
plt.show()

# Return
count = np.count_nonzero(resnet_image != preprocessed_augmented_image)
print('Number of different values between image and image put through augmentation layers (no augmentations applied):', count)

