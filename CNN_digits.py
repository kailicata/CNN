import numpy as np 
import keras
import json
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from testing_CNN import Conv_Neural_Net

debug_training = False
do_training = False
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

if debug_training == True:
    x_train = x_train[0:200,:,:]
    y_train = y_train[0:200]


if do_training == True:
    CNN = Conv_Neural_Net()
    x_train_cat, y_train_cat = CNN.prepare_data(x_train,y_train)
    CNN.build_model()
    history = CNN.fit(x_train_cat, y_train_cat)
    CNN.save_model("CNN_model_trial_zero.h5")
    with open("training_history.json", "w") as f:
        json.dump(history.history, f)
else: 
    CNN = Conv_Neural_Net()
    CNN.load("CNN_model_trial_zero.h5")
    x_test_cat, y_test_cat = CNN.test(x_test, y_test)
    with open("training_history.json", "r") as f:
        history = json.load(f)

    score = CNN.model.evaluate(x_test_cat, y_test_cat, verbose=0)
    print("Test loss: ", score[0])
    print("Test accuracy: ", score[1])


    plt.plot(history['loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Model Loss Over Epochs')
    plt.show()
















