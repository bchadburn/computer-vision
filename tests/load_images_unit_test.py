import unittest
import sys
import os

# Importing modules from parent folder
current_dir = os.getcwd()
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from utils.dataset import Dataset
from utils.config import *

batch_size = 32  # For tests below, exact value doesn't matter


class TestLoadImages(unittest.TestCase):
    def setUp(self):
        self.Dataset = Dataset(os.path.join(parent_dir, TRAINING_IMAGES_PATH), batch_size, IMAGE_SIZE)
        self.Dataset.classes = ALL_CLASSES
        self.image_filenames, self.image_targets = self.Dataset._load_images()

    def test_same_length(self):
        self.assertEqual(len(self.image_filenames), len(self.image_targets))

    def test_correct_format(self):
        prefix_list = [str(i) + '-' for i in self.Dataset.classes]
        for i in self.image_filenames:
            self.assertIn(i.split(os.path.sep)[1][0:2], prefix_list)
            self.assertTrue(i.endswith('.jpg'))

    def test_image_targets(self):
        classes = [str(i) for i in ALL_CLASSES]
        for i in self.image_targets:
            self.assertIn(i, classes)


if __name__ == '__main__':
    print("Unit test function _load_images from training.py script")
    print("training images file path:", TRAINING_IMAGES_PATH)
    unittest.main()
