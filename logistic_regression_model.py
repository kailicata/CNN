import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1/(1+np.exp(-x))

class LogisticRegression():


    def __init__(self, lr=0.001, n_iters=1000):
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None
        self.list_of_total_error = []
        self.list_of_total_num_steps = []
        
    def save(self, model_file):
        np.save(model_file, self.weights)
        print("model saved")

    def load(self, model_file):
        self.weights = np.load(model_file)
        print("model loaded")

    def fit(self, X,y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        print("number of iterations" + str(self.n_iters))

        
        

        for i in range(self.n_iters):
            linear_pred = np.dot(X, self.weights) + self.bias
            predictions = sigmoid(linear_pred)

            #calcultating the change in error, this is the derivative of cross entopy 
            #error is a vector 
            error = predictions - y
            dw = (1/n_samples) * np.dot(X.T, (error))
            db = (1/n_samples) * np.sum(predictions - y)

            #total error
            total_error = np.sum(np.abs((error)))
            print("iteration " + str(i) + " sum of total error " + str(total_error))

            if i%10 == 0:
                self.list_of_total_error.append(total_error)
                self.list_of_total_num_steps.append(i)


            print(" ")

            #average of total error
            average_total_error = np.sum(np.abs((error)))/len(X)
            #print("iteration " + str(i) + " average error " + str(average_total_error))

            print(" ")
            print(" ")


            print("the type of dw " + str(dw.shape))
            self.weights = self.weights - self.lr*dw
            self.bias = self.bias - self.lr*db

            #print("weights: " + str(i) + str(self.weights))
            #print(str(type(dw)))

        print(" weight: "+  str(self.weights))






    def predict(self, X):
        linear_pred = np.dot(X, self.weights) + self.bias
        y_pred = sigmoid(linear_pred)
        class_pred = [0 if y<=0.5 else 1 for y in y_pred]
        return class_pred

