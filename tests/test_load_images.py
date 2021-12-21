import pytest
import os
import sys
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
from indoor_outdoor.utils.constants import TRAINING_IMAGES_PATH, IMAGE_SIZE
from indoor_outdoor.utils.config import ALL_CLASSES
from indoor_outdoor.dataset import Dataset

batch_size = 32


@pytest.fixture
def dataset():
    dataset = Dataset(TRAINING_IMAGES_PATH, batch_size, IMAGE_SIZE)
    dataset.classes = ALL_CLASSES
    image_filenames, image_targets = dataset._load_images()
    return dataset, image_filenames, image_targets


def test_same_length(dataset):
    dataset, image_filenames, image_targets = dataset
    assert len(image_filenames) == len(image_targets)


def test_correct_format(dataset):
    dataset, image_filenames, image_targets = dataset
    prefix_list = [str(i) + '-' for i in dataset.classes]
    for i in image_filenames:
        assert i.split(os.path.sep)[1][0:2] in prefix_list
        assert i.endswith('.jpg')


def test_image_targets(dataset):
    dataset, image_filenames, image_targets = dataset
    classes = [str(i) for i in ALL_CLASSES]
    for i in image_targets:
        assert i in classes
