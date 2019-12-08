import keras
import keras.backend as K
from keras.models import Sequential
from keras.layers import Flatten, Dense


class Model(Sequential):
    def __init__(self, *args):
        super().__init__()
        units = list(args)

        self.add(Flatten())
        for i, unit in enumerate(units):
            if i == 0:
                self.add(Dense(unit, input_dim=28 * 28))
            elif i == len(units) - 1:
                self.add(Dense(unit, activation="sigmoid"))
            else:
                self.add(Dense(unit))
