import unittest
from utils import Dataset
from utils.config import *

test1 = {"product": {"id": 870752, "title": "Caviar Quilted Small Double Flap", "colors": [{"id": 'null', "name": 'null'}, {"id": 9, "name": "Blue"}, {"id": 58, "name": "Black"}, {"id": 109, "name": "Chartreuse"}, {"id": 386, "name": "Beige"}, {"id": 649, "name": "White"}, {"id": 871, "name": "Green"}, {"id": 1025, "name": "Purple"}, {"id": 1080, "name": "Dark Beige"}, {"id": 1178, "name": "Pink"}, {"id": 1244, "name": "Orange"}, {"id": 1295, "name": "Dark Brown"}, {"id": 1477, "name": "Light Blue"}, {"id": 1552, "name": "Red"}, {"id": 1640, "name": "Beige Clair"}, {"id": 1678, "name": "Rose"}, {"id": 1742, "name": "Dark Grey"}, {"id": 1768, "name": "Light Pink"}, {"id": 1963, "name": "Light Brown"}, {"id": 2126, "name": "Navy"}, {"id": 2180, "name": "Light Purple"}, {"id": 2309, "name": "Grey"}, {"id": 2612, "name": "Dark Pink"}, {"id": 2949, "name": "Yellow"}, {"id": 3573, "name": "Light Green"}, {"id": 4180, "name": "Burgundy"}], "colorId": 58, "styleId": 100591, "imageUrl": "https://www.fashionphile.com/images/product-images/thumb/5d2d9c35e4b088cc8b072b9302187576/7e316a0a362daeb438e6ed8422145865.jpg", "colorName": "Black"}, "metaData": {"brand_id": 95, "quote_id": 2105764, "calibration": {"version": 7, "features": {"score": 0.397027314, "recall": 0.527363184079602, "precision": 0.4568965517241379, "recall_recent": 0.5513626834381551, "support_recent": 954, "weighted_score": 0.3565361235, "precision_recent": 0.6102088167053364}}, "brand_version": "resnet", "text_prediction": {"size": "SMALL", "rules": ["return_final_size"], "colors": "WHITE ", "rules_off": [], "style_colors": "YELLOW,PURPLE,LIGHT BLUE,GREY,BURGUNDY,DARK PINK,RED,BEIGE CLAIR,BLUE,BEIGE,NAVY,PINK,WHITE,DARK RED,BLACK", "style_parent_colors": "YELLOWS,PURPLES,GRAYS,BEIGE,BLUES,PINK,WHITES,REDS,BLACK"}, "container_version": "202107126be8", "filtering_version": "2101", "calibration_enabled": True, "groundtruth_style_id": 'null', "customer_provided_name": "Small double flap white ", "groundtruth_style_name": 'null', "text_prediction_enabled": True}}
test1['metaData']
batch_size = 32  # For tests below, exact value doesn't matter
test['tier']
test['quoteData']['inference']['tier']
class TestLoadImages(unittest.TestCase):
    def setUp(self):
        self.Dataset = Dataset(TRAINING_IMAGES_PATH, batch_size, IMAGE_SIZE)
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
