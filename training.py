from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import numpy as np
from preprocessing import preprocess_data

# Paths to training and testing datasets
TRAIN_PATH = "C:/Users/skart/Downloads/QA/TRAIN"  # Ensure this path is correct
TEST_PATH = "C:/Users/skart/Downloads/QA/TEST"    # Ensure this path is correct

IMG_SIZE = 224
NUM_CLASSES = 2  # Adjust based on the number of classes
EPOCHS = 10
BATCH_SIZE = 32

def create_model(input_shape, num_classes):
    """
    Create and compile the CNN model.
    """
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Dropout(0.2),
        
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Dropout(0.3),
        
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.4),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# Load and preprocess the datasets
X_train, y_train, class_names = preprocess_data(TRAIN_PATH, NUM_CLASSES)
X_test, y_test, _ = preprocess_data(TEST_PATH, NUM_CLASSES)

# Create the model
model = create_model((IMG_SIZE, IMG_SIZE, 3), NUM_CLASSES)

# Train the model
history = model.fit(X_train, y_train,
                    validation_split=0.2,
                    epochs=EPOCHS,
                    batch_size=BATCH_SIZE,
                    verbose=1)

# Evaluate the model
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Accuracy: {test_accuracy:.2f}")

# Generate predictions on the test set
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)  # Convert predictions to class labels
y_true_classes = np.argmax(y_test, axis=1)  # Convert true labels to class labels

# Print classification report (Precision, Recall, F1-Score)
print("\nClassification Report:")
print(classification_report(y_true_classes, y_pred_classes, target_names=class_names))

# Confusion Matrix
cm = confusion_matrix(y_true_classes, y_pred_classes)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.savefig("C:/Users/skart/Downloads/QA/confusion_matrix.png")
plt.show()

# Misclassification Analysis (optional, only if you want to inspect some examples)
misclassified_idx = np.where(y_true_classes != y_pred_classes)[0]
print(f"\nNumber of misclassified samples: {len(misclassified_idx)}")

# Display some misclassified images
for idx in misclassified_idx[:5]:  # Display first 5 misclassified images
    plt.imshow(X_test[idx])
    plt.title(f"True: {class_names[y_true_classes[idx]]}, Pred: {class_names[y_pred_classes[idx]]}")
    plt.axis('off')
    plt.show()

# Save the model
model.save("C:/Users/skart/Downloads/QA/oral_disease_model.keras")
print("Model saved as C:/Users/skart/Downloads/QA/oral_disease_model.keras")

# Plot training history
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title('Accuracy Over Epochs')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title('Loss Over Epochs')
plt.savefig("C:/Users/skart/Downloads/QA/training_history.png")
