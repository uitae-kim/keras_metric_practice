import keras
import keras.backend as K
from keras.models import Sequential
from keras.layers import Flatten, Dense
import utils


class Model:
    def __init__(self, *args):
        units = list(args)

        self.model = Sequential()
        self.model.add(Flatten())
        for i, unit in enumerate(units):
            if i == 0:
                self.model.add(Dense(unit, input_dim=784, activation='relu'))
            elif i == len(units)-1:
                self.model.add(Dense(unit, activation="sigmoid"))
            else:
                self.model.add(Dense(unit))

        # 안에선 컴파일하는 케이스로 해봅시다
        self.model.compile(optimizer='adam', loss=utils.f1_geometric_loss, metrics=['accuracy', utils.p_metric, utils.r_metric, utils.f1_geometric])

    def fit(self, x, y):
        self.model.fit(x, y)

    def evaluate(self, x, y):
        return self.model.evaluate(x, y)
