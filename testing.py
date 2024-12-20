import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import load_model
from preprocessing import preprocess_data

TEST_PATH = "C:/Users/skart/Downloads/QA/TEST"
MODEL_PATH = "C:/Users/skart/Downloads/QA/oral_disease_model.keras"
CLASS_NAMES = ["Caries", "Gingivitis"]

model = load_model(MODEL_PATH)
X_test, y_test, _ = preprocess_data(TEST_PATH, num_classes=2)

test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Accuracy: {test_accuracy:.2f}")

def predict_and_visualize(model, images, labels, class_names, num_samples=20):
    plt.figure(figsize=(15, 10))
    for i in range(min(num_samples, len(images))):
        img = images[i]
        true_label = class_names[np.argmax(labels[i])]
        pred_probs = model.predict(img[np.newaxis, ...])
        pred_label = class_names[np.argmax(pred_probs)]
        plt.subplot(4, 5, i + 1)
        plt.imshow(img)
        plt.title(f"True: {true_label}\nPred: {pred_label}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()

predict_and_visualize(model, X_test, y_test, CLASS_NAMES, num_samples=20)
