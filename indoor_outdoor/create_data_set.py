import pandas as pd
from pathlib import Path
import shutil
import json
import argparse
import os
from utils.config import INDOOR_SCENES, OUTDOOR_SCENES, INDOOR_LABEL, OUTDOOR_LABEL
from utils.constants import RAW_FILES_PATH, RAW_DATA_PATH, TRAINING_IMAGES_PATH
from utils.utils import create_directory


def map_classes(metadata, indoor_classes, outdoor_classes):
    indoor_matches = metadata['Name'].apply(lambda x: True if x in indoor_classes else False)
    outdoor_matches = metadata['Name'].apply(lambda x: True if x in outdoor_classes else False)
    # Return indices for words found in indoor and outdoor class lists.
    mapping_index = {'indoor': list(metadata['Index'][indoor_matches]), 'outdoor': list(metadata['Index'][outdoor_matches])}
    return mapping_index


def map_parent_category(loc_mapping, list_image_info):
    map_image_location = {}
    for image_dict in list_image_info:
        tmp_labels = set(image_dict['labels'])

        if tmp_labels.intersection(loc_mapping['indoor']):
            map_image_location[image_dict['long_id']] = INDOOR_LABEL

        if tmp_labels.intersection(loc_mapping['outdoor']):
            map_image_location[image_dict['long_id']] = OUTDOOR_LABEL

    return map_image_location


def move_images(orig_path, dest_path, image_dict):
    for file in Path(orig_path).glob('*.jpg'):
        image_id = Path(file).stem

        if image_id in image_dict.keys():
            file_name = f'{image_dict[image_id]}-{image_id}.jpg'
            final_path = os.path.join(dest_path, file_name)
            shutil.copy(file, final_path)


def main(image_dest):
    if not os.path.exists(args.image_destination):
        os.makedirs(args.image_destination)

    vocab = pd.read_csv(os.path.join(RAW_FILES_PATH, 'vocabulary.csv'))
    f = open(os.path.join(RAW_FILES_PATH, 'video_category_data.json'))
    list_image_details = json.load(f)

    create_directory(image_dest)
    location_mapping = map_classes(vocab, INDOOR_SCENES, OUTDOOR_SCENES)
    image_locations = map_parent_category(location_mapping, list_image_details)

    # Delete any images currently in source directory
    print('Deleting existing files in image destination')
    for file in os.listdir(image_dest):
        os.remove(os.path.join(image_dest, file))

    image_source_path = os.path.join(RAW_DATA_PATH)
    move_images(image_source_path, image_dest, image_locations)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_destination', type=str, default=TRAINING_IMAGES_PATH)
    args, _ = parser.parse_known_args()

    main(args.image_destination)
    print('Completed creating data set')
