import os

# Image path settings
DATA_DIRECTORY = 'data'
IMAGE_DIRECTORY = 'indoor_outdoor_images'
RAW_FILES_PATH = os.path.join(DATA_DIRECTORY, 'indoor_outdoor_raw')
RAW_DATA_PATH = os.path.join(RAW_FILES_PATH, 'images')
TRAINING_IMAGES_PATH = os.path.join(DATA_DIRECTORY, IMAGE_DIRECTORY)

# Model Settings
IMAGE_SIZE = 224  # Image size required by modeling. Resnet uses 224.
MODEL_RESULTS_PATH = os.path.join(os.getcwd(), 'model_results')  # Path for confusion matrix and image predictions, parent folder for .pb modeling file
MODEL_CHECKPOINT_PATH = os.path.join(os.getcwd(), 'model_checkpoint')
