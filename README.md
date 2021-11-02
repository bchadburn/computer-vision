# Indoor-Outdoor Image Classification

![alt text](https://www.csun.edu/sites/default/files/AS-Earth_Month-Outdoor_Online.jpg)

This is an image classification project using labelled images from the video dataset from YouTube.
The purpose is to classify indoor and outdoor scenes with limited data and efficient model training
practices. Unit test examples are provided, along with code to return single model predictions. 

## Getting Started
Download images, video_category_data.json, and vocabulary.csv

2 options: 
1. Download subset of images from an AZURE storage container, video_category_data.json and vocabulary.csv
https://dresearch.blob.core.windows.net/public-indoor-outdoor/indoor_outdoor.zip
2. Skip create_data_set step: Download file "indoor_outdoor_images". This folder contains 
169 curated images. 
   - The images were selected from the AZURE subset of images using the following classes
     - indoor_scenes = {'Bedroom', 'Bathroom', 'Classroom', 'Office', 'Living Room', 'Dining Room', 'Room'
     - outdoor_scenes = {'Landscape', 'Skyscraper', 'Mountain', 'Beach', 'Ocean'}
   - Images that were incorrectly labelled as outdoor, indoor were removed. No hard/easy images were removed or added
    
**1st method**: Ensure parent folder "indoor_outdoor" is in project director. 
Within this folder include a folder "images". Also, within the parent folder include the two files: video_category_data.json and vocabulary.csv
If using this method, after creating dataset, you may want to review images as some
completely blank and several are mislabelled. These curated images are provided in the zip file "indoor_outddoor_images."

**2nd method**: Unzip and place folder "indoor_outdoor_images" in project directory and skip step: Creating Data Set.

Along with the image related files, ensure you have the following scripts in project director.
* requirements.txt file
* environment.yml (for conda users)
* utils folder with utils.py, preprocess_image.py, config.py
* create_data_set.py
* Model folder with model.py
* training.py
* single_image_predictions.py
* load_images_unit_test.py

## Installation

- Create a python virtual environment in the system and activate it.

**Installation using pip:**
  - `pip install virtualenv`
  - `virtualenv <env_name>`
  - `source <env_name>/bin/activate`

Install the dependencies for the project using the requirements.txt
  - `pip install -r requirements.txt`

**Installation for conda users:**

The packages may fail to load if using installing from requirements.txt file as conda-forge may be required to download certain packages.
Instead, use environment.yml file.
- conda env create --name <env_name> --file=environment.yml
- conda activate <env_name>

### Configuration of image folders, classes, and model
config.py includes settings for destination directories, image, and model settings. No changes should be needed unless
changing target classes or reorganizing image directories

Image path settings
* PARENT_DIRECTORY: Parent folder with image folder, video_category_data.json, and vocabulary.csv
* IMAGE_DIRECTORY: change original image folder name
* TRAINING_IMAGES_PATH: modify folder for training images. Can also be changed from CLI when running 
  create_data_set.py and training.py
  
Class settings
* indoor/outdoor_label: Sets labels for target classes
* ALL_CLASSES: Set list of classes of target classes
* PRED_CLASS_NAMES: Labels corresponding to prediction index (e.g. "indoor")
* indoor/outdoor scenes lists related classes grouped as "indoor", "outdoor"

Model Settings
* MODEL_RESULTS_PATH: Path for confusion matrix and image predictions, parent folder for .pb model file
* IMAGE_SIZE: Image size required by model. Resnet uses 244.
* MODEL_CHECKPOINT_PATH: Path for saving model checkpoints

Note: Any changes to target classes will require changes to map_classes and map_parent_category under the [Creating data set section](#Creating-data-set)

### Creating data set
Run python create_data_set.py from CLI. The default destination of images is
"indoor_outdoor_images". 

To pass a different path run: python create_data_set.py --image_destination <path_to_folder>
If you pass a different path, you will need to pass the new image source path when running
training.py (see [Training section](#Training) for details).

#### Description: 
Script uses video_category_data.json to map images to specific labels. It uses vocabulary.csv
to then map the labels to 'class' names listed in the variables indoor_scenes and outdoor_scenes. 
Relevant images are then moved to a new folder and are given a prefix to indicate whether the image should 
be indoor or outdoor (e.g. "0-" to specify an indoor image, "1-" for outdoor).

##### Important functions
* map_classes: Maps Indoor, Outdoor to relevant 'class' labels. If other classes or wanted, 
  the function will need to be modified.
* map_parent_category: maps images to indoor, outdoor classes. If other classes or wanted, 
  the function will need to be modified.

### Training
Run python training.py

The default source of training images is indoor_outdoor_images. To specify source run:
python training.py --image_path <path_to_folder>
Other optional parameters can be passed for model training including:
- --epochs:  The number of epochs that will be used to train the initial classification model.
- --learning_rate: The learning rate that will be used to train the model
- --fine_tuning_epochs: The number of epochs that will be used to fine tune the model. If zero is specified, the model will not 
                                go through the fine-tuning process
- --fine_tuning_learning_rate:  The learning rate that will be used to fine tune the model.   

#### Description
Script trains a CNN model by adding layers on top of a pre-trained ResNet model. During fine-tuning stage, it unfreezes all layers and finishes training model.
Outputs confusion matrix, and model predictions and confidence scores for specific images. Returns overall model 
performance on validation dataset, and lastly outputs trained model.

Troubleshooting note: To run script, ensure system is using numpy version specified in requirements.txt file for numpy version. 
Script can throw the error "NotImplementedError: Cannot convert a symbolic Tensor..." for some newer versions of numpy.

### Making single image predictions
Run python single_image_predictions -i {image path}  
For example, if passing images used for train/val: python single_image_prediction.py -i indoor_outdoor_images/0-_2hRjVpJtdY.jpg

#### Description
Loads model and makes prediction on provided image. Prints predicted class and confidence score.
 
### Run Tensorflow GPU vs CPU test
When running training.py file, the script will run a tensor through a simple CNN layer 
through CPU and will try using GPU and compare test times. The results will be printed in the console. 

If tensorflow-gpu is correctly configured, the GPU should be considerably faster, 
although its dependent on GPU. Using NVIDIA GeForce RTX 2080 Super, the GPU speed over CPU is 600+% for this test.

### Run _load_image unit test
Run python load_images_unit_test.py 

Note: Test script is meant to support being run automatically and doesn't take arguments. If the training images 
aren't found in the default folder 'indoor_outdoor_images', the variable TRAINING_IMAGES_PATH needs to be changed in 
config.py

#### Description
Tests result of _load_images function from training.py script. The function _load_images returns the filename and target class list.
The script runs the following three tests:
1. Number of filenames match number of target classes. 
2. Image format includes prefix class + '-'  (e.g. '0-'), and the extension is .jpg
3. Checks target classes are found in class list defined in config.py file

### Model Improvements

**Correcting Image Labels:** Initially 183 indoor and outdoor images were found and moved to the training folder. The images
were reviewed to ensure labels were correctly assigned. 14 images were deleted as they were bad images (some blank, some were irrelevant). A few images 
labelled as indoor were outdoor images and visa-versa, so the labels were changed accordingly. 

**Model Performance:** After curation, the model's performance fluctuates between 95-100%. 
Small model architecture and parameter tweaking were needed to improve from 94% to high 90s but otherwise, the model has high performance 
even if the top layers are changed a bit. However,
the results can vary a few percent between trainings. 

**Next Steps:** To further improve the solution, I probably would enquire about the business requirement for accuracy, reliability, and speed and whether more improvement is needed. Typically, a final test set would also need to be created from images
we haven't seen and that properly reflect images the model would see in production.

**Model Improvements:** Otherwise, to improve the model further I would focus on a 'data centric' approach. 

1. Perform another round of curation and make sure no other bad images were missed. 

2. Identify the type of mistakes the model is making and how that differs
between trainings. For that reason, predictions for each image were exported after training.
   
3. Add other indoor/outdoor categories (similar to building, house etc) where images could be added or find other options to add more data. 
   It's likely to have a more reliable model, more images and examples would be useful given the high variability between images and low validation count.
   
4. Review data augmentations: With more time. I would be helpful to review different data augmentation methods and values. Given the 
   variability in our images, more aggressive augmentation could be added.

5. Hyper-parameter tuning, and fiddling with different architectures could also be useful, but I'd first start with 
business requirements and understanding the current limitations of the existing model. 
If model improvements are warranted, I'd probably use Keras Tuner to help efficiently explore the hyper-parameter search space for model parameters
and augmentation and fine-tune parameters from there. Finally, stacked and/or ensemble methods could be used to try getting optimum performance.

Speed: Given the low number of images, the model training takes place in under a minute. Since we are using gpu enabled tf, tf.data instead of slower methods like
Keras generators, parallel processing for image processing and loading, prefetch images in CPU while GPU runs etc. 
the model training is capable of training efficiently on a much larger dataset. Also, by default TF decodes to jpeg 
using "Integer Fast" method. If "Integer Accurate" method is needed, decoding may be slightly slower but due to prefetch
decrease in training time should be minimal.