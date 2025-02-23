import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt
from logistic_regression_model import LogisticRegression



bc = datasets.load_breast_cancer()
X, y = bc.data, bc.target
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state=1234)
print("training data" + str(len(X_train)))
print("testing data" + str(len(X_test)))



lr1 = 0.001
clf1 = LogisticRegression(lr=lr1)
clf1.fit(X_train, y_train)
y_pred = clf1.predict(X_test)
clf1.save("logistic_cancer_model1")

lr2 = 0.00001
clf2 = LogisticRegression(lr=lr2)
clf2.fit(X_train, y_train)
y_pred = clf2.predict(X_test)
clf2.save("logistic_cancer_model2")

def accuracy(y_pred, y_test):
    return np.sum(y_pred==y_test)/len(y_test)

acc = accuracy(y_pred, y_test)
print(acc)









# Plot the values
plt.plot(clf1.list_of_total_num_steps, clf1.list_of_total_error, marker='o', linestyle='-',color='blue', label=f"Learning Rate {lr1}")
plt.plot(clf2.list_of_total_num_steps, clf2.list_of_total_error, marker='s', linestyle='--',color='red', label=f"Learning Rate {lr2}")

# Labels and title
plt.xlabel("Iteration Number ")
plt.ylabel("Train Loss")
plt.title("Comparison of Learning Rates")

# Add legend with the specified labels
plt.legend()

# Show the plot
plt.show()





