# Indoor-Outdoor Image Classification

Binary image classifier (indoor vs. outdoor scenes) trained on a curated subset of YouTube-8M
frames using transfer learning on ResNet-50 with TensorFlow.

## Approach

- **Dataset**: 169 curated images from YouTube-8M covering 12 scene categories
  (Bedroom, Bathroom, Classroom, Office, Living Room, Dining Room for indoor;
  Landscape, Skyscraper, Mountain, Beach, Ocean for outdoor)
- **Model**: ResNet-50 backbone (ImageNet pretrained) with fine-tuned classification head
- **Techniques**: Image augmentation, early stopping, efficient training on limited data
- **Testing**: Pytest suite covering data loading and augmentation pipelines

## Project Structure

```
indoor_outdoor/
├── model.py                    # ResNet-50 model definition
├── training.py                 # Training loop
├── single_image_predictions.py # Inference on individual images
└── utils/
    ├── config.py
    ├── dataset.py
    ├── image_functions.py
    └── utils.py
data/                           # Image data (not committed)
references/                     # indoor_outdoor_images.zip (169 curated images)
tests/                          # Pytest suite
```

## Quickstart

Requires Python 3.9+ and [uv](https://github.com/astral-sh/uv).

```bash
git clone https://github.com/bchadburn/computer-vision.git
cd computer-vision
uv venv && source .venv/bin/activate
uv pip install -r requirements.txt
```

Unzip the curated image set:
```bash
unzip references/indoor_outdoor_images.zip -d data/
```

Train the classifier:
```bash
python indoor_outdoor/training.py
```

Run inference on a single image:
```bash
python indoor_outdoor/single_image_predictions.py --image path/to/image.jpg
```

## Run Tests

```bash
pytest tests/ -v
```
