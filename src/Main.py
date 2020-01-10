from NeuralNetwork import NeuralNetwork
from Utils import load_input
import numpy as np
import matplotlib.pyplot as plt

dataset = load_input('../datasets/data.h5')
X_train, Y_train = dataset["X_train"]/255*2 - 1, dataset["Y_train"]
X_test, Y_test = dataset["X_test"]/255*2 - 1, dataset["Y_test"]

model = NeuralNetwork()
model.add_layer(20, 'relu', inputs=X_train.shape[0])
model.add_layer(7, 'relu')
model.add_layer(5, 'relu')
model.add_layer(1, "sigmoid")
model.compile(learning_rate=0.05, regularization='L2', lambd=0.01)

costs = model.fit(X_train, Y_train, 1000, verbose=True, step=10)

plt.plot(costs)
plt.show()

prediction_train = model.predict(X_train)
prediction_test = model.predict(X_test)

print("train accuracy: {} %".format(100 - np.mean(np.abs(prediction_train - Y_train)) * 100))
print("test accuracy: {} %".format(100 - np.mean(np.abs(prediction_test - Y_test)) * 100))