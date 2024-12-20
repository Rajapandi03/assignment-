import os
import cv2
import numpy as np
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models

IMG_SIZE = 224  # The target image size (224x224 pixels for models like VGG, ResNet, etc.)
NUM_CLASSES = 2  # Number of classes (Caries, Gingivitis)

def load_data(dataset_path):
    """
    Load images and their labels from the dataset directory.
    
    dataset_path: str
        The path to the main dataset folder that contains subfolders for each class.
        
    Returns:
        images (numpy array): Array of processed images.
        labels (numpy array): Array of labels corresponding to each image.
        class_names (list): List of class folder names (e.g., 'Caries', 'Gingivitis').
    """
    images = []  # List to store image data
    labels = []  # List to store corresponding class labels
    class_names = os.listdir(dataset_path)  # Get the class names (subfolder names)

    for label, class_name in enumerate(class_names):
        class_folder = os.path.join(dataset_path, class_name)  # Path to the current class folder
        
        # Loop through all image files in the class folder
        for img_file in os.listdir(class_folder):  # Loop through each image file
            img_path = os.path.join(class_folder, img_file)  # Full path to the image file
            img = cv2.imread(img_path)  # Read the image using OpenCV
            
            if img is not None:  # Check if image is valid
                # Resize image and convert to RGB format
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))  
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                # Append the processed image and its label to the lists
                images.append(img)
                labels.append(label)
    
    # Return the images and labels as numpy arrays
    return np.array(images), np.array(labels), class_names

# Load the training and testing datasets
train_path = r"C:\Users\skart\Downloads\QA\TRAIN"
test_path = r"C:\Users\skart\Downloads\QA\TEST"

X_train, y_train, class_names = load_data(train_path)
X_test, y_test, _ = load_data(test_path)

print(f"Loaded {len(X_train)} training images from {len(class_names)} classes.")
print(f"Loaded {len(X_test)} testing images.")

# Normalize the images
X_train = X_train / 255.0
X_test = X_test / 255.0

# One-hot encode the labels
y_train = to_categorical(y_train, NUM_CLASSES)
y_test = to_categorical(y_test, NUM_CLASSES)

# Data augmentation using ImageDataGenerator
datagen = ImageDataGenerator(
    rotation_range=15,       # Rotate images up to 15 degrees
    width_shift_range=0.1,   # Shift images horizontally by 10%
    height_shift_range=0.1,  # Shift images vertically by 10%
    horizontal_flip=True,    # Randomly flip images horizontally
)

datagen.fit(X_train)

# Define a simple CNN model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(NUM_CLASSES, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model using data augmentation
history = model.fit(datagen.flow(X_train, y_train, batch_size=32), epochs=10, validation_data=(X_test, y_test))

# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_acc * 100:.2f}%")
