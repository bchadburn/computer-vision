import os

sep = os.path.sep

# Image path settings
PARENT_DIRECTORY = 'indoor_outdoor'
IMAGE_DIRECTORY = 'images'
IMAGE_SOURCE_PATH = os.path.join(PARENT_DIRECTORY, IMAGE_DIRECTORY)
TRAINING_IMAGES_PATH = 'indoor_outdoor_images'

# Class settings
INDOOR_LABEL = 0
OUTDOOR_LABEL = 1
ALL_CLASSES = (0, 1)
INDOOR_SCENES = ('Bedroom', 'Bathroom', 'Classroom', 'Office', 'Living Room', 'Dining Room', 'Room')
OUTDOOR_SCENES = ('Landscape', 'Skyscraper', 'Mountain', 'Beach', 'Ocean')
PRED_CLASS_NAMES = ("outdoor", "indoor")

# Model Settings
IMAGE_SIZE = 224  # Image size required by model. Resnet uses 224.
MODEL_RESULTS_PATH = 'model_results'  # Path for confusion matrix and image predictions, parent folder for .pb model file
MODEL_CHECKPOINT_PATH = 'model_checkpoint'
