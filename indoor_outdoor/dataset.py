import tensorflow as tf
import os
from sklearn import model_selection
from utils.image_functions import parse_image_to_tensor
from utils.constants import IMAGE_SIZE
from utils.config import ALL_CLASSES


class Dataset:
    def __init__(self, dataset_directory, batch_size, image_size):
        self.dataset_directory = dataset_directory
        self.batch_size = batch_size
        self.image_size = image_size

        self.classes = None
        self.total_images = None
        self.class_weight = None

        self.train_dataset = None
        self.validation_dataset = None

        self.validation_metadata = None

    def load(self, classes=None):
        """Creates training and validation dataset from directory of images. The prefix of the image name is used to
        identify the class."""

        # Assigns class based on image name prefix (e.g. 0-_2hRjVpJtdY.jpg is assigned to class 0 i.e. 'indoor')
        if not classes:
            self.classes = list(set(
                file.split('-')[0] for file in os.listdir(self.dataset_directory)))
            self.classes.sort()

        print(f"Classes: {len(self.classes)}. {self.classes}")
        all_classes_str = map(str, ALL_CLASSES)
        diff_list = set(self.classes) - set(all_classes_str)
        if len(diff_list) > 0:
            raise ValueError(f"Classes in image directory don't match classes defined in config.py file. "
                             f"Class differences: {diff_list}")

        image_filenames, image_targets = self._load_images()

        (
            image_filenames_train,
            image_filenames_validation,
            image_targets_train,
            image_targets_validation
        ) = model_selection.train_test_split(
            image_filenames,
            image_targets,
            train_size=0.8,
            stratify=image_targets,
            shuffle=True,
            random_state=42
        )

        self.train_dataset = self._dataset(image_filenames_train, image_targets_train,
                                           self.batch_size, repeat=True)
        self.validation_dataset = self._dataset(image_filenames_validation, image_targets_validation,
                                                self.batch_size, repeat=False)
        self.validation_metadata = self._dataset(image_filenames_validation, image_targets_validation, batch_size=0,
                                                 repeat=False, metadata=True)

    def _load_images(self):
        """Loads image filenames and their respective classes. Calculated total images and class weights in case of
        class imbalance."""
        image_filenames = []

        for name in os.listdir(self.dataset_directory):
            if name.split('-')[0] in self.classes:
                image_filenames.append(os.path.join(self.dataset_directory, name))

        self.total_images = len(image_filenames)
        image_filenames = sorted(image_filenames)

        # Get class target from image name prefix
        image_targets = [self.classes.index(os.path.split(name)[1].split('-')[0]) for name in image_filenames]

        # Create dict with each class and their respective image count as a % of total images
        self.class_weight = dict((i, image_targets.count(i) / len(image_targets)) for i in image_targets)

        return image_filenames, image_targets

    def _dataset(self, image_filenames, image_targets, batch_size, repeat=False, metadata=False):
        """Creates tf.data tensors of filenames, targets used for training. Sets up parallel processing
        for image loading and parsing. Sets prefetch to load next batch in CPU, while GPU is in training."""
        image_filenames_dataset = tf.data.Dataset.from_tensor_slices(image_filenames)

        target_dataset = tf.data.Dataset.from_tensor_slices(image_targets)
        image_dataset = image_filenames_dataset if metadata else image_filenames_dataset.map(
            lambda x: parse_image_to_tensor(x, IMAGE_SIZE),
            num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = tf.data.Dataset.zip((image_dataset, target_dataset))

        if batch_size > 0:  # batch_size is 0 when creating dataset of the image filenames
            dataset = dataset.batch(batch_size)

        if repeat:
            dataset = dataset.repeat()

        # Prefetch data in CPU, while GPU is training
        dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

        return dataset
