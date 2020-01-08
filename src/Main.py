from NeuralNetwork import NeuralNetwork
from Utils import load_input
import numpy as np

dataset = load_input('../datasets/data.h5')
X_train, Y_train = dataset["X_train"]/255*2 - 1, dataset["Y_train"]
X_test, Y_test = dataset["X_test"]/255*2 - 1, dataset["Y_test"]

model = NeuralNetwork()
model.add_layer(20, 'relu', inputs=X_train.shape[0])
model.add_layer(7, 'relu')
model.add_layer(5, 'relu')
model.add_layer(1, "sigmoid")
model.learning_rate = 0.05

model.fit(X_train, Y_train, 1000, verbose=True)

prediction_train = model.predict(X_train)
prediction_test = model.predict(X_test)

print("train accuracy: {} %".format(100 - np.mean(np.abs(prediction_train - Y_train)) * 100))
print("test accuracy: {} %".format(100 - np.mean(np.abs(prediction_test - Y_test)) * 100))