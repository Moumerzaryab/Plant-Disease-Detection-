Plant Disease Prediction with CNN
Overview
This project implements a Convolutional Neural Network (CNN) to predict plant diseases from images using TensorFlow. The model is trained and tested on the New Plant Diseases Dataset from Kaggle, which contains images of various plant leaves categorized into 38 classes representing healthy and diseased states across multiple plant species.
The project consists of two main Jupyter Notebooks:

Train_plant_disease.ipynb: Builds, trains, and evaluates the CNN model.
Test_plant_disease.ipynb: Loads the trained model to perform predictions on single images and visualize results.

Dataset

Source: New Plant Diseases Dataset on Kaggle.
Description:
Training Set: 70,295 images across 38 classes.
Validation Set: 17,572 images across 38 classes.
Classes: Includes diseases and healthy states for plants like Apple, Tomato, Corn, Grape, and more (e.g., Apple___Apple_scab, Tomato___healthy).
Image Format: RGB images resized to 128x128 pixels.


Preprocessing:
Images are loaded using tf.keras.utils.image_dataset_from_directory with a batch size of 32.
Labels are one-hot encoded for multi-class classification.
No data augmentation is applied in the provided code, but it can be added for improved robustness.



Model Architecture
The CNN model is built using TensorFlow's Keras API with the following architecture:

Input: RGB images of size 128x128 pixels.
Layers:
Five Convolutional Blocks: Each block contains:
Two Conv2D layers with increasing filters (32, 64, 128, 256, 512), 3x3 kernels, ReLU activation, and 'same' padding for the first layer.
A MaxPool2D layer (2x2 pool size, stride 2) to reduce spatial dimensions.


Dropout: Applied after convolutional blocks (0.25) and dense layer (0.4) to prevent overfitting.
Flattening: Converts convolutional outputs to a 1D vector.
Dense Layer: 1500 neurons with ReLU activation.
Output Layer: 38 neurons with softmax activation for multi-class classification.


Parameters: Approximately 7.8 million, primarily due to the large dense layer.
Compilation:
Optimizer: Adam with a learning rate of 0.0001 to avoid overshooting.
Loss Function: Categorical Crossentropy.
Metric: Accuracy.


Evaluation: A confusion matrix is generated to visualize performance on the validation set.

Prerequisites
To run the notebooks, you need the following dependencies:

Python 3.9+
TensorFlow
NumPy
Matplotlib
Seaborn
Pandas

Install the dependencies using:
pip install tensorflow numpy matplotlib seaborn pandas

Additionally, ensure you have the New Plant Diseases Dataset downloaded and organized in the following structure:
project_directory/
├── train/
│   ├── Apple___Apple_scab/
│   ├── Apple___healthy/
│   └── ...
├── valid/
│   ├── Apple___Apple_scab/
│   ├── Apple___healthy/
│   └── ...
├── Train_plant_disease.ipynb
├── Test_plant_disease.ipynb
├── 2trained_plant_disease_model2.keras

Usage

Training the Model:

Open Train_plant_disease.ipynb in Jupyter Notebook or Google Colab.
Ensure the dataset is placed in the train and valid directories.
Run the notebook to preprocess the data, build, and train the CNN model.
The trained model will be saved as 2trained_plant_disease_model2.keras.


Testing the Model:

Open Test_plant_disease.ipynb in Jupyter Notebook or Google Colab.
Ensure the pre-trained model (2trained_plant_disease_model2.keras) and validation dataset are available.
Run the notebook to load the model and predict the disease for a single image.
The notebook will display the image with the predicted disease name (e.g., Tomato___Late_blight).


Example Prediction:To predict on a single image, you can add the following code to Test_plant_disease.ipynb:
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Load class names from validation set
validation_set = tf.keras.utils.image_dataset_from_directory(
    'valid', labels='inferred', label_mode='categorical', image_size=(128, 128), batch_size=32)
class_name = validation_set.class_names

# Load model
cnn = tf.keras.models.load_model('2trained_plant_disease_model2.keras')

# Load and preprocess image
image_path = 'valid/Tomato___Late_blight/0003faa8-4b62-4b66-89cc-9b0db24e17b4___GHLB2 Leaf 8917.JPG'
img = tf.keras.preprocessing.image.load_img(image_path, target_size=(128, 128))
img_array = tf.keras.preprocessing.image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0) / 255.0  # Normalize

# Predict
predictions = cnn.predict(img_array)
result_index = np.argmax(predictions[0])
model_prediction = class_name[result_index]

# Visualize
plt.imshow(img)
plt.title(f"Disease Name: {model_prediction}")
plt.xticks([])
plt.yticks([])
plt.show()



Results

The model is trained to classify 38 plant disease classes with high accuracy (specific accuracy depends on training epochs and data).
A confusion matrix is generated in Train_plant_disease.ipynb to visualize classification performance.
The testing notebook successfully predicts and visualizes the disease for a single image, as shown in the output plot.

Potential Improvements

Data Augmentation: Add techniques like random flips, rotations, and zooms to improve model robustness.data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.2),
    tf.keras.layers.RandomZoom(0.2)
])


Transfer Learning: Use pre-trained models (e.g., ResNet50, EfficientNet) to improve performance and reduce training time.
Batch Evaluation: Evaluate the model on the entire validation set to report accuracy and generate a confusion matrix.
Confidence Scores: Display prediction probabilities for the top classes during testing.
GPU Acceleration: Use GPU support for faster training, as the current setup indicates no GPU availability.

Contributing
Contributions are welcome! To contribute:

Fork the repository.
Create a new branch (git checkout -b feature-branch).
Make your changes and commit (git commit -m "Add feature").
Push to the branch (git push origin feature-branch).
Create a pull request.

Please ensure your code follows the existing style and includes appropriate documentation.
License
This project is licensed under the MIT License. See the LICENSE file for details.
Authors

Moumer Zaryab
Ansar Hayat

Acknowledgments

Thanks to the creators of the New Plant Diseases Dataset for providing the data.
Built with TensorFlow, Matplotlib, and other open-source libraries.

