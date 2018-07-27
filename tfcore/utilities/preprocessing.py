"""
Copyright (C) Silvio Jurk - All Rights Reserved
Unauthorized copying of this file, via any medium is strictly prohibited
Proprietary and confidential.

@author: Silvio Jurk
"""

import abc
import cv2 as cv
import scipy
from tfcore.utilities.image import *


class Preprocessing():

    def __init__(self):
        self.functions = ([], [], [])

    def add_function_x(self, user_function):

        self.functions[0].append(user_function)

    def add_function_y(self, user_function):

        self.functions[1].append(user_function)

    def add_function_xy(self, user_function):

        self.functions[2].append(user_function)

    def run(self, img_x, img_y):

        for func in self.functions[2]:
            img_x, img_y = func(img_x, img_y)

        for func in self.functions[0]:
            img_x, _ = func(img_x, None)

        for func in self.functions[1]:
            _, img_y = func(None, img_y)

        return img_x, img_y

    class Base():
        def __init__(self, modes=[], shuffle=True):
            self.shuffle = shuffle
            self.modes = modes
            self.idx = 0

        @property
        def new_index(self):
            if self.shuffle:
                self.idx = randint(0, len(self.modes) - 1)
            else:
                self.idx += 1
                if self.idx >= len(self.modes) - 1:
                    self.idx = 0

            return self.idx

        @abc.abstractmethod
        def function(self, img_x, img_y):
            raise NotImplementedError("Please Implement this method")

        def function_iterator(self, iterator):
            for image_x, image_y in iterator:
                img_x, img_y = self.function(image_x, image_y)
                yield img_x, img_y

    class Scale(Base):

        def __init__(self, factors=(2, 3, 4), interp='bicubic', shuffle=True):
            self.interp = interp
            super().__init__(modes=factors, shuffle=shuffle)

        def function(self, img_x, img_y):
            shape = img_x.shape
            index = self.new_index
            img_x = resize(img_x, (int(shape[0] / self.modes[index]), int(shape[1] / self.modes[index])), self.interp)
            img_x = resize(img_x, (int(shape[0]), int(shape[1])), self.interp)

            return img_x, img_y

    class DownScale(Base):

        def __init__(self, factors=2, interp='bicubic'):
            self.interp = interp
            super().__init__(modes=[factors])

        def function(self, img_x, img_y):
            img_x = resize(img_x, (int(img_x.shape[0] / self.modes[0]), int(img_x.shape[1] / self.modes[0])), self.interp)
            return img_x, img_y

    class Flip(Base):

        def __init__(self, direction=('horizontal', 'vertical'), shuffle=True):
            super().__init__(modes=direction, shuffle=shuffle)

        def function(self, img_x, img_y):
            index = self.new_index
            if img_x is not None:
                img_x = cv.flip(img_x, index)
            if img_y is not None:
                img_y = cv.flip(img_y, index)
            return img_x, img_y

    class Rotate(Base):

        def __init__(self, angle=(), steps=10, shuffle=True):
            if len(angle) == 0:
                angle = [steps * i for i in range(360 // steps)]
            super().__init__(modes=angle, shuffle=shuffle)

        def function(self, img_x, img_y):
            index = self.new_index
            if img_x is not None:
                img_x = scipy.ndimage.rotate(img_x, self.modes[index], reshape=False, prefilter=False, mode='reflect')
            if img_y is not None:
                img_y = scipy.ndimage.rotate(img_y, self.modes[index], reshape=False, prefilter=False, mode='reflect')
            return img_x, img_y

    class Brightness(Base):

        def __init__(self, shuffle=True):
            super().__init__(modes=(), shuffle=shuffle)

        def function(self, img_x, img_y):
            min = np.min(img_x)
            max = abs(256 - np.max(img_x))
            value = randint(0, max + min) - min

            if img_x is not None:
                img_x = np.clip(img_x + value, 0, 255)
            if img_y is not None:
                img_y = np.clip(img_y + value, 0, 255)
            return img_x, img_y
