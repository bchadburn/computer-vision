from utils.image_functions import *
import argparse
from utils.utils import dir_file
from utils.config import *


if __name__ == "__main__":
    """Returns predictions for a single image"""
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", dest="image_path", required=True,
                        help="Full path to image, must be .jpg", metavar="FILE", type=lambda x: dir_file(x))
    args, _ = parser.parse_known_args()

    if not args.image_path.endswith(".jpg"):
        if not args.image_path.endswith(".jpeg"):
            raise TypeError("Invalid file to return model predictions, file needs to end with .jpg")

    image = process_single_image(args.image_path, IMAGE_SIZE)
    fine_tuned_model = tf.keras.models.load_model(os.path.join(MODEL_RESULTS_PATH, "fine_tuned_model"))

    prediction = fine_tuned_model.predict(image)
    pred_index = np.argmax(prediction, axis=1)
    prediction = round(prediction[0][pred_index][0], 4)
    print('class: ', PRED_CLASS_NAMES[pred_index[0]])
    print('score: ', f'{prediction:.2%}')
