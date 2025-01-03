# Dogs vs Cats Classification

## Project Description
This project aims to classify images of dogs and cats using a deep learning model built with TensorFlow. The model is trained on a dataset of images and uses convolutional neural networks (CNNs) to achieve classification. The dataset is provided by Kaggle's Dogs vs. Cats competition.

The project includes dataset preparation, neural network construction, training, and model evaluation. The model is then used for prediction on unseen images, showcasing the power of deep learning in image classification tasks.

## Technology Used
- **Python**: Programming language used for implementing the solution.
- **TensorFlow**: Deep learning framework used for building and training the neural network.
- **Keras**: High-level neural networks API for building and training models.
- **Matplotlib**: Used for plotting training and validation accuracy and loss graphs.
- **Jupyter Notebook**: For running and testing the code interactively.

### How it is Implemented:
1. **Dataset Preparation**:
   - Download the dataset from Kaggle’s [Dogs vs Cats competition](https://www.kaggle.com/competitions/dogs-vs-cats/data)
   - Unzip the dataset and rename the directory to `dogs-vs-cats-original`.
   - Organize the dataset into three subsets: training, validation, and test datasets.

2. **Data Pipeline**:
   - Use `image_dataset_from_directory()` to create TensorFlow dataset objects for training, validation, and testing.

3. **Model Construction**:
   - A Convolutional Neural Network (CNN) is designed with convolutional layers and pooling layers.
   - The model is compiled using an optimizer, loss function, and metrics.

4. **Training**:
   - The model is trained using the `fit()` function, and performance metrics (accuracy, loss) are plotted.
   - A custom model name is saved after training using the format `model.{first-name}-{last-name}.{something-that-you-can-identify-what-this-model-is}.keras`.

5. **Prediction**:
   - The trained model is used for making predictions on new images.
   - The image is displayed, and the prediction result is printed.

## Repository Organization

### Dataset Folder Preparation
1. **Download** the `dogs-vs-cats.zip` file from Kaggle: [Dogs vs Cats Dataset](https://www.kaggle.com/competitions/dogs-vs-cats/data).
2. **Unzip** the file and rename the directory as `dogs-vs-cats-original`.

### Python Implementation - `dogs_cats.py`
Contains the core class `DogsCats` which implements all the following functions:

1. **`__init__(self)`**: Initializes datasets (train, validation, test) and model as `None`.
2. **`make_dataset_folders(self, subset_name, start_index, end_index)`**: Creates dataset folders for train, validation, and test.
3. **`_make_dataset(self, subset_name)`**: Creates TensorFlow dataset objects using `image_dataset_from_directory()` for train, valid, and test datasets.
4. **`make_dataset(self)`**: Calls `_make_dataset()` to generate train, validation, and test datasets.
5. **`build_network(self, augmentation=True)`**: Builds a neural network using convolutional layers and pooling layers. Optionally, applies data augmentation.
6. **`train(self, model_name)`**: Trains the model and saves it as a `.keras` file with a custom model name.
7. **`load_model(self, model_name)`**: Loads a trained model.
8. **`predict(self, image_file)`**: Makes predictions on a given image file and displays the result.

### Jupyter Notebook - `module.ipynb`
- Demonstrates the implementation of the `DogsCats` class from `dogs_cats.py`.
- Shows how to organize the dataset into training, validation, and test sets:
    - 0 - 2,399: Validation
    - 2,400 - 11,999: Training
    - 12,000 - 12,499: Testing
- Creates TensorFlow dataset objects using the `make_dataset()` method.
- Builds and trains the neural network using `build_network()` and `train()`.
- Displays model summary and visualizes training results.

## Hyperparameters
- **CLASS_NAMES**: ['dog', 'cat']
- **IMAGE_SHAPE**: (180, 180, 3)
- **BATCH_SIZE**: 32
- **BASE_DIR**: pathlib.Path('dogs-vs-cats')
- **SRC_DIR**: pathlib.Path('dogs-vs-cats-original/train')
- **EPOCHS**: 20


## How to Run the Project
1. Clone the repository.
2. Download the dataset from Kaggle and prepare the dataset folder.
3. Install required dependencies:
    ```bash
    pip install tensorflow matplotlib
    ```
4. Run the Jupyter Notebook `module.ipynb` to interact with the code and visualize the results.

## Conclusion
This project demonstrates how to implement an image classification model using TensorFlow and Keras, as well as how to organize, train, and evaluate a deep learning model for real-world tasks like image classification.

