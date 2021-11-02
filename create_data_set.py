import pandas as pd
from pathlib import Path
import shutil
import json
import argparse
from utils.config import *
from utils.utils import create_directory

image_source = f'{PARENT_DIRECTORY}/{IMAGE_DIRECTORY}'


def map_classes(metadata, indoor_classes, outdoor_classes):
    mapping_index = {'indoor': [], 'outdoor': []}
    for index, row in metadata.iterrows():
        if row['Name'] in indoor_classes:
            mapping_index['indoor'].append(row['Index'])
        if row['Name'] in outdoor_classes:
            mapping_index['outdoor'].append(row['Index'])
    return mapping_index


def map_parent_category(location_mappings):
    map_image_location = {}
    for i in image_labels:
        if any(l in location_mappings['indoor'] for l in i['labels']):
            map_image_location[i['long_id']] = INDOOR_LABEL
        if any(l in location_mappings['outdoor'] for l in i['labels']):
            map_image_location[i['long_id']] = OUTDOOR_LABEL
    return map_image_location


def move_images(orig_path, dest_path, image_dict):
    for file in Path(orig_path).glob("*.jpg"):
        image_id = str(file.resolve()).split('\\')[-1].replace('.jpg', '')
        if image_id in image_dict.keys():
            file_name = str(image_dict[image_id]) + '-' + image_id + '.jpg'
            final_path = os.path.join(dest_path, file_name)
            shutil.copy(file, final_path)


def main(image_dest):
    create_directory(image_dest)
    location_classes = map_classes(vocab, INDOOR_SCENES, OUTDOOR_SCENES)
    image_locations = map_parent_category(location_classes)
    move_images(image_source, image_dest, image_locations)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_destination", type=str, default=TRAINING_IMAGES_PATH)
    args, _ = parser.parse_known_args()

    if not os.path.exists(args.image_destination):
        os.makedirs(args.image_destination)

    vocab = pd.read_csv(f'{PARENT_DIRECTORY}/vocabulary.csv')
    f = open(f'{PARENT_DIRECTORY}/video_category_data.json')
    image_labels = json.load(f)

    main(args.image_destination)
    print("Completed creating data set")
