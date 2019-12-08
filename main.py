import keras
from model import Model

data = keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = data.load_data()
model = Model(25, 10)
model.fit(x_train, y_train)
print(model.evaluate(x_test, y_test))
