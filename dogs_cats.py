import tensorflow as tf
import pathlib
import os
import shutil
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models, optimizers, losses, callbacks

class DogsCats:
    """Class to handle dataset preparation, model building, training, and prediction for the Dogs-Cats classification"""

    CLASS_NAMES = ['dog', 'cat']
    IMAGE_SHAPE = (180, 180, 3)
    BATCH_SIZE = 32
    BASE_DIR = pathlib.Path('dogs-vs-cats')
    SRC_DIR = pathlib.Path('dogs-vs-cats-original/train')
    EPOCHS = 20

    def __init__(self):
        """Initialize datasets and model as None"""
        self.train_dataset = None
        self.valid_dataset = None
        self.test_dataset = None
        self.model = None

    def make_dataset_folders(self,subset_name, start_index, end_index):
        """Create dataset folders for training, validation, and testing subsets"""
        for category in ("dog", "cat"):
            dir = self.BASE_DIR / subset_name / category
            if os.path.exists(dir) is False:
                os.makedirs(dir)
            files = [f'{category}.{i}.jpg' for i in range(start_index, end_index)]
            for i, file in enumerate(files):
                shutil.copyfile(src=self.SRC_DIR / file, dst=dir / file)
                if i % 100 == 0:
                    print(f'src:{self.SRC_DIR / file} => dst:{dir / file}')

    def _make_dataset(self, subset_name):
        """Create a TensorFlow Dataset object from directory"""
        dataset = tf.keras.preprocessing.image_dataset_from_directory(
            self.BASE_DIR / subset_name,
            image_size=self.IMAGE_SHAPE[:2],
            batch_size=self.BATCH_SIZE,
            label_mode='binary'
        )
        return dataset

    def make_dataset(self):
        """train, validation, and test datasets"""
        self.train_dataset = self._make_dataset('train')
        self.valid_dataset = self._make_dataset('valid')
        self.test_dataset = self._make_dataset('test')

    # def build_network(self, augmentation=True):
    #     """Build and compile a CNN"""
    #     model = models.Sequential()

    #     if augmentation:
    #         data_augmentation = tf.keras.Sequential([
    #             layers.RandomFlip('horizontal'),
    #             layers.RandomRotation(0.1),
    #             layers.RandomZoom(0.2)
    #         ])
    #         model.add(data_augmentation)

    #     inputs = layers.Input(shape=(180, 180, 3))
    #     x = data_augmentation(inputs)
    #     x = layers.Rescaling(1./255)(x)
    #     x = layers.Conv2D(filters=32, kernel_size=3, activation="relu")(x)
    #     x = layers.AveragePooling2D(pool_size=2)(x)
    #     x = layers.Conv2D(filters=64, kernel_size=3, activation="relu")(x)
    #     x = layers.AveragePooling2D(pool_size=2)(x)
    #     x = layers.Conv2D(filters=128, kernel_size=3, activation="relu")(x)
    #     x = layers.AveragePooling2D(pool_size=2)(x)
    #     x = layers.Conv2D(filters=256, kernel_size=3, activation="relu")(x)
    #     x = layers.AveragePooling2D(pool_size=2)(x)

    #     x = layers.Flatten()(x)
    #     x = layers.Dense(200, activation="relu")(x)
    #     outputs = layers.Dense(1, activation="sigmoid")(x)
    #     model = tf.keras.Model(inputs=inputs, outputs=outputs)

    # def build_network(self, augmentation=True):
    #     model = models.Sequential()

    #     # if augmentation:
    #     #     model.add(layers.experimental.preprocessing.RandomFlip('horizontal'))
    #     #     model.add(layers.experimental.preprocessing.RandomRotation(0.2))
    #     if augmentation:
    #         data_augmentation = tf.keras.Sequential([
    #             layers.RandomFlip('horizontal'),
    #             layers.RandomRotation(0.1),
    #             layers.RandomZoom(0.2)
    #         ])
    #         model.add(data_augmentation)

    #     model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=self.IMAGE_SHAPE))
    #     model.add(layers.MaxPooling2D((2, 2)))
    #     model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    #     model.add(layers.MaxPooling2D((2, 2)))
    #     model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    #     model.add(layers.MaxPooling2D((2, 2)))
    #     model.add(layers.Flatten())
    #     model.add(layers.Dense(128, activation='relu'))
    #     model.add(layers.Dense(1, activation='sigmoid'))

    #     # Compile the model
    #     model.compile(
    #         optimizer=optimizers.Adam(),
    #         loss=losses.BinaryCrossentropy(),
    #         metrics=['accuracy']
    #     )
    #     self.model = model

    def build_network(self, augmentation=True):
        """Build and compile a CNN model."""
        model = models.Sequential()

        # Data augmentation layers
        if augmentation:
            data_augmentation = tf.keras.Sequential([
                layers.RandomFlip('horizontal'),
                layers.RandomRotation(0.1),
                layers.RandomZoom(0.2)
            ])
            model.add(data_augmentation)

        # Add convolutional and pooling layers
        model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=self.IMAGE_SHAPE))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(128, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))

        # Fully connected layers
        model.add(layers.Flatten())
        model.add(layers.Dense(128, activation='relu'))
        model.add(layers.Dense(1, activation='sigmoid'))

        # Compile the model
        model.compile(
            optimizer=optimizers.Adam(),
            loss=losses.BinaryCrossentropy(),
            metrics=['accuracy']
        )

        # Assign to the class attribute
        self.model = model


    def train(self, model_name):
        """Train the model"""
        checkpoint_cb = callbacks.ModelCheckpoint(
            filepath=model_name,
            save_best_only=True
        )
        early_stopping_cb = callbacks.EarlyStopping(patience=5, restore_best_weights=True)

        history = self.model.fit(
            self.train_dataset,
            validation_data=self.valid_dataset,
            epochs=self.EPOCHS,
            callbacks=[checkpoint_cb, early_stopping_cb]
        )

        # Plot accuracy and loss
        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']
        loss = history.history['loss']
        val_loss = history.history['val_loss']

        plt.figure(figsize=(8, 4))
        plt.subplot(1, 2, 1)
        plt.plot(acc, label='Training Accuracy')
        plt.plot(val_acc, label='Validation Accuracy')
        plt.legend()
        plt.title('Accuracy')

        plt.subplot(1, 2, 2)
        plt.plot(loss, label='Training Loss')
        plt.plot(val_loss, label='Validation Loss')
        plt.legend()
        plt.title('Loss')
        plt.show()

    def load_model(self, model_name):
        """Load the pre-trained model"""
        self.model = tf.keras.models.load_model(model_name)

    def predict(self, image_file):
        """Predict the class of a given image"""
        img = tf.keras.utils.load_img(image_file, target_size=self.IMAGE_SHAPE[:2])
        img_array = tf.keras.utils.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0) / 255.0
        prediction = self.model.predict(img_array)[0][0]
        predicted_class = 'dog' if prediction > 0.5 else 'cat'

        plt.imshow(img)
        plt.title(f"Prediction: {predicted_class} ({prediction:.2f})")
        plt.axis('off')
        plt.show()
