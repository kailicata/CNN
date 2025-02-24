import numpy as np 
import keras
import json
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from testing_CNN import Conv_Neural_Net
from PIL import Image


def preprocess_image(image_path):
    img = Image.open(image_path).convert("L")  # Convert to grayscale
    img = img.resize((28, 28))
    img_array = np.array(img)
    img_array = img_array / 255.0
    img_array = img_array.reshape(1, 28, 28, 1)  # (1 sample, 28x28, 1 channel)
    
    plt.imshow(img, cmap="gray")
    plt.axis("off")
    plt.title("Processed Image (28x28 Grayscale)")
    plt.show()
 

    return img_array

visualizing_keras_image = False

#loading files
CNN = Conv_Neural_Net()  # Initialize your CNN class
CNN.load("CNN_model_trial_zero.h5")  # Load your trained model
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()


if visualizing_keras_image == True:
    plt.figure(figsize=(10, 2))  # Adjust figure size
    for i in range(10):
        plt.subplot(2, 5, i + 1)  # Create a 2-row, 5-column grid
        plt.imshow(x_train[i], cmap="gray")  # Display in grayscale
        plt.axis("off")  # Hide axis
        plt.title(f"Label: {y_train[i]}")  # Show corresponding digit label

    plt.show()
else:
    image_path = "/Users/kailicata/Desktop/thinseven.png"
    processed_image = preprocess_image(image_path)

    prediction = CNN.model.predict(processed_image)

    # Get the predicted class (digit)
    predicted_digit = np.argmax(prediction)
    print(f"Predicted Digit: {predicted_digit}")




