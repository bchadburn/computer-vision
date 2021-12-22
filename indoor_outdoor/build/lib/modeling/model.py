from tensorflow.keras import layers, Model, Sequential
from tensorflow.keras.applications import resnet_v2
from tensorflow.keras.layers.experimental import preprocessing


def create_model(num_classes, image_size):
    """Loads ResNet modeling (v2) with imagenet weights. Includes data augmentation layers inside modeling for GPU support."""
    data_augmentation = Sequential([
        preprocessing.RandomFlip("horizontal", input_shape=(image_size, image_size, 3), seed=42),
        preprocessing.RandomRotation(factor=(-0.2, 0.2), seed=42),
        preprocessing.RandomZoom(height_factor=(-0.2, 0.2), width_factor=(-0.2, 0.2), seed=42),
        preprocessing.RandomTranslation(height_factor=(-0.1, 0.1), width_factor=(-0.1, 0.1), seed=42),
    ])

    base_model = resnet_v2.ResNet50V2(
        weights='imagenet',
        include_top=False
    )

    base_model.trainable = False

    model_inputs = layers.Input(shape=(image_size, image_size, 3), name="image")
    x = data_augmentation(model_inputs)
    x = resnet_v2.preprocess_input(x)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.8)(x)
    model_outputs = layers.Dense(num_classes, activation='softmax')(x)
    model = Model(name="classification", inputs=model_inputs, outputs=model_outputs)

    return model
