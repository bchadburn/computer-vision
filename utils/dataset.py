import numpy as np
import os
import tensorflow as tf
from sklearn import model_selection
from utils.image_functions import _parse_image
from utils.config import *


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

        tmp_classes = classes if classes else set(
            [' '.join(i[0].split('-')) for i in os.listdir(self.dataset_directory)])
        self.classes = list(tmp_classes)

        print(f"Classes: {len(self.classes)}. {self.classes}")

        all_classes_str = [str(i) for i in tmp_classes]
        diff_list = list(set(all_classes_str) - set(tmp_classes))
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
        for classname in self.classes:
            for name in os.listdir(self.dataset_directory):
                if name.split('-')[0] == classname:
                    image_filenames.append(os.path.join(self.dataset_directory, name))

        self.total_images = len(image_filenames)
        image_filenames = sorted(image_filenames)

        image_targets = [self.classes.index(name.split("\\")[1].split('-')[0]) for name in image_filenames]

        self.class_weight = dict(zip(np.arange(len(image_targets)), 1.0 / np.bincount(image_targets)))
        return image_filenames, image_targets

    def _dataset(self, image_filenames, image_targets, batch_size, repeat=False, metadata=False):
        """Creates tf.data tensors of filenames, targets used for training. Sets up parallel processing
        for image loading and parsing. Sets prefecth to load next batch in CPU, while GPU is in training."""
        image_filenames_dataset = tf.data.Dataset.from_tensor_slices(image_filenames)

        target_dataset = tf.data.Dataset.from_tensor_slices(image_targets)
        image_dataset = image_filenames_dataset if metadata else image_filenames_dataset.map(
            lambda x: _parse_image(x, IMAGE_SIZE),
            num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = tf.data.Dataset.zip((image_dataset, target_dataset))

        # batch_size is 0 when creating dataset of the image filenames, otherwise a batch of images is created for training
        if batch_size > 0:
            dataset = dataset.batch(batch_size)

        if repeat:
            dataset = dataset.repeat()

        # Prefetch data in CPU, while GPU is training
        dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

        return dataset