import pandas as pd
import numpy as np
import argparse
import os
from sklearn import metrics
from tensorflow.keras import callbacks
from tensorflow.keras import optimizers

from modeling import model as model_utils
from utils.utils import verify_create_paths, time_run, dir_path
from utils.constants import TRAINING_IMAGES_PATH, IMAGE_SIZE, MODEL_RESULTS_PATH, MODEL_CHECKPOINT_PATH
from dataset import Dataset
from utils.gpu_config import check_cpu_gpu, gpu, cpu


def evaluate_model(session, model, dataset):
    """Returns predictions on validation data set along with image names to review specific predictions. Prints
    classification report, saves confusion matrix and predictions for each filename."""
    y_true = list(dict(dataset.validation_metadata.as_numpy_iterator()).values())
    validation_predictions = model.predict(dataset.validation_dataset)
    y_pred = np.argmax(validation_predictions, axis=1)

    print(metrics.classification_report(
        y_true,
        y_pred,
        target_names=dataset.classes
    ))

    confusion_matrix = pd.DataFrame(
        metrics.confusion_matrix(y_true, y_pred),
        columns=dataset.classes,
        index=dataset.classes
    )

    confusion_matrix.to_csv(os.path.join(MODEL_RESULTS_PATH, f'{session}-confusion-matrix.csv'))

    # Create list of image files to match to validation predictions during evaluation
    full_file_names = list(dict(dataset.validation_metadata.as_numpy_iterator()).keys())
    # Splits string to return original image name e.g. '0-2LvJQGVwqms.jpg' becomes '2LvJQGVwqms.jpg' as the prefix '0-' defines the class
    image_names = list(map(lambda s: os.path.split(s.decode())[-1], full_file_names))

    predictions = pd.DataFrame({
        'filename': image_names,
        'target': np.array(dataset.classes)[y_true],
        'prediction': np.array(dataset.classes)[y_pred],
        'confidence': np.max(validation_predictions, axis=1)
    })

    predictions.to_csv(os.path.join(MODEL_RESULTS_PATH, f'{session}-predictions.csv'))

    accuracy = metrics.accuracy_score(y_true, y_pred)
    print(f'Validation accuracy: {accuracy}')
    return accuracy


def train_model(data, model, checkpoint_path, learn_rate, num_epochs):
    """Compiles and trains model"""
    model.compile(
        optimizer=optimizers.Adam(learning_rate=learn_rate),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    print(model.summary())

    # Save the weights of the best modeling
    model_checkpoint = callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        save_best_only=True,
        save_weights_only=True,
        monitor='val_loss',
        mode='min',
    )


    model.fit(
        data.train_dataset,
        validation_data=data.validation_dataset,
        epochs=num_epochs,
        steps_per_epoch=data.total_images / data.batch_size,
        callbacks=[model_checkpoint],
        class_weight=data.class_weight,
        verbose=2
    )

    model.load_weights(checkpoint_path)
    return model


@time_run
def fit_model(
        model,
        dataset,
        learning_rate,
        epochs,
        fine_tuning_learning_rate,
        fine_tuning_epochs
):
    """Compiles and trains modeling with modeling checkpoint. Returns performance on entire validation set.
     If fine-tuning epochs>0, it runs fine-tuning where all layers are unfrozen.
     Finally, it loads best weights, and returns performance on validation set."""
    current_dir = os.getcwd()  # Windows requires full path or doesn't have required permissions to access cp.ckpt
    checkpoint_filepath = os.path.join(current_dir, MODEL_CHECKPOINT_PATH)
    checkpoint_filepath = os.path.join(checkpoint_filepath, 'cp.ckpt')
    os.makedirs(checkpoint_filepath, exist_ok=True)

    model = train_model(dataset, model, checkpoint_filepath, learning_rate, epochs)
    print('Classification modeling completed training.')

    accuracy = evaluate_model('training', model, dataset)

    if accuracy == 1 or fine_tuning_epochs == 0:
        print('Fine tuning classification modeling skipped.')
        if accuracy == 1:
            print(f'Accuracy reached {accuracy:.0%}')
        return model

    for layer in model.layers:
        layer.trainable = True

    model = train_model(dataset, model, checkpoint_filepath, fine_tuning_learning_rate, fine_tuning_epochs)
    print('Classification modeling completed fine-tunining.')

    evaluate_model('fine-tuning', model, dataset)

    return model


def main(image_path, learning_rate, epochs, fine_tuning_learning_rate, fine_tuning_epochs, batch_size):
    check_cpu_gpu()

    verify_create_paths(MODEL_RESULTS_PATH, MODEL_CHECKPOINT_PATH)

    # Create filename, target datasets, and filename tensors in prep for modeling training
    dataset = Dataset(image_path, batch_size=batch_size, image_size=IMAGE_SIZE)
    dataset.load(classes=None)

    model = model_utils.create_model(len(dataset.classes), IMAGE_SIZE)

    model = fit_model(
        model,
        dataset,
        learning_rate,
        epochs,
        fine_tuning_learning_rate,
        fine_tuning_epochs
    )

    model.save(os.path.join(MODEL_RESULTS_PATH, 'fine_tuned_model'))

    print('Model was successfully saved.')


if __name__ == '__main__':
    '''Builds dataset from image files, creates and fits modeling and saves modeling weights.'''
    parser = argparse.ArgumentParser()
    parser.add_argument('-training_images_path', type=lambda x: dir_path(x), default=os.path.join(os.getcwd(), TRAINING_IMAGES_PATH),
                        help='Full path to training images folder')

    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='The learning rate that will be used to train the classification modeling.')

    parser.add_argument('--epochs', type=int, default=15,
                        help='The number of epochs that will be used to train the initial classification modeling.')

    parser.add_argument('--batch_size', type=int, default=32,
                        help='The batch size used for training.')

    parser.add_argument('--fine_tuning_learning_rate', type=float, default=1e-5,
                        help='The learning rate that will be used to fine tune the classification modeling.')

    parser.add_argument('--fine_tuning_epochs', type=int, default=15,
                        help=(
                                'The number of epochs that will be used to fine tune the classification modeling. '
                                'If zero is specified, the modeling will skip fine tuning.')
                        )
    args, _ = parser.parse_known_args()

    main(args.training_images_path, args.learning_rate, args.epochs,
         args.fine_tuning_learning_rate, args.fine_tuning_epochs, args.batch_size)
