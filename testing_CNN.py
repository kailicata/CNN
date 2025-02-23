import numpy as np 
import keras
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import load_model





class Conv_Neural_Net:
    def __init__(self,num_classes=10, input_shape=(28,28,1)):
        self.num_classes = num_classes
        self.input_shape = input_shape 
        self.model = None
    


    def prepare_data(self,x_train,y_train):
        x_train = x_train.astype("float32")/255  #dvidiing by 255 to make number between 0 and 1 
        x_train_cat = np.expand_dims(x_train, -1) #exapmding last dimension to have a depth of 1 
        print("x_train shape: ", x_train_cat.shape)
        #print(x_train.shape[0], "train samples")


        y_train_cat = keras.utils.to_categorical(y_train, self.num_classes) # converting class vectors to binary class matricies 
        print("y train shape" + str(y_train_cat.shape))



        return x_train_cat, y_train_cat
        

    def test(self,x_test, y_test):
        x_test = x_test.astype("float32")/255
        x_test_cat = np.expand_dims(x_test, -1)
        print(x_test.shape[0], "test samples")

        y_test_cat = keras.utils.to_categorical(y_test, self.num_classes)

        return x_test_cat, y_test_cat



    def build_model(self):
        self.model = keras.Sequential(
            [
                keras.Input(shape=self.input_shape),
                layers.Conv2D(32, kernel_size=(3,3),activation="relu"),
                layers.MaxPooling2D(pool_size=(2,2)),
                layers.Conv2D(64, kernel_size=(3,3),activation="relu"),
                layers.MaxPooling2D(pool_size=(2,2)),
                layers.Flatten(),
                layers.Dropout(0.5),
                layers.Dense( self.num_classes, activation="softmax"),
            ]
        )
        self.model.summary()


   

    def save_model(self,CNN_model_file):
        self.model.save(CNN_model_file)
        print("model saved")

    def load(self,CNN_model_file):
        self.model = load_model(CNN_model_file)
        print("Model loaded")



    def fit(self, x_train_cat, y_train_cat):
        batch_size = 128
        epochs = 15

        self.model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
        history = self.model.fit(x_train_cat, y_train_cat, batch_size=batch_size, epochs=epochs, validation_split=0.1)

        return history 

    






"""
CNN = Conv_Neural_Net()
CNN.build()


score = CNN.model.evaluate(x_test, y_test, verbose=0)
print("Test loss: ", score[0])
print("Test accuracy: ", score[1])
"""







