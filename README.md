Facial Emotion Recognition using VGG16 and Keras Sequential Model

Overview

This project develops a deep learning model to recognize emotions in facial images using VGG16 and Keras' Sequential API. The model is trained on a dataset of facial expressions to classify emotions such as happy, sad, angry, surprised, and neutral.

Features

Use of pre-trained VGG16 for feature extraction

Custom Sequential model for fine-tuned classification

Data preprocessing including normalization and augmentation

Training and evaluation of the model using performance metrics

Visualization of predictions and results

Dataset

The dataset consists of labeled facial images categorized into different emotions. It includes:

Happy

Sad

Angry

Surprised

Neutral

Technologies Used

Python

TensorFlow/Keras

OpenCV

NumPy

Pandas

Scikit-learn

Matplotlib/Seaborn

Installation

Clone the repository:

git clone https://github.com/yourusername/facial-emotion-recognition.git
cd facial-emotion-recognition

Install dependencies:

pip install -r requirements.txt

Model Architecture

The model consists of:

VGG16 as a feature extractor – Pre-trained on ImageNet, used for extracting deep features.

Custom Sequential layers – Fully connected layers added on top for emotion classification.

Activation Functions – ReLU for hidden layers and Softmax for output.

Loss Function – Categorical Crossentropy for multi-class classification.

Optimizer – Adam optimizer for efficient training.

Usage

Train the model:

python train.py

Evaluate the model:

python evaluate.py

Predict emotions from images:

python predict.py --image path/to/image.jpg

Visualize predictions:

python visualize.py

Results

The model achieves competitive accuracy in recognizing emotions from facial images. Performance is evaluated using:

Accuracy

Precision, Recall, and F1-score

Confusion Matrix

Sample predictions with visualization

Future Improvements

Fine-tuning VGG16 layers for better accuracy

Experimenting with other architectures like ResNet or EfficientNet

Adding real-time emotion detection using OpenCV

Training on a larger, more diverse dataset

License

This project is licensed under the MIT License.
