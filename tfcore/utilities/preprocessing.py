import abc
import cv2 as cv
import matplotlib.pyplot as plt
import scipy
from skimage.measure import regionprops
from tfcore.utilities.image import *
from PIL import Image


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
            if img_x is not None:
                img_x = resize(img_x, (int(shape[0] / self.modes[index]), int(shape[1] / self.modes[index])), self.interp)

            if img_y is not None:
                img_y = resize(img_y, (int(shape[0]), int(shape[1])), self.interp)

            return img_x, img_y

    class DownScale(Base):

        def __init__(self, factors=2, interp='bicubic'):
            self.interp = interp
            super().__init__(modes=[factors])

        def function(self, img_x, img_y):
            if img_x is not None:
                img_x = resize(img_x, (int(img_x.shape[0] / self.modes[0]), int(img_x.shape[1] / self.modes[0])), self.interp)

            if img_y is not None:
                img_y = resize(img_y, (int(img_y.shape[0] / self.modes[0]), int(img_y.shape[1] / self.modes[0])), self.interp)

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
            #cv.imshow('image_in', img_x)
            #cv.waitKey(0)
            #cv.imshow('image_in', img_y)
            #cv.waitKey(0)
            index = self.new_index
            if img_x is not None:
                img_x = scipy.ndimage.rotate(img_x, self.modes[index], reshape=False, prefilter=False, mode='reflect')
            if img_y is not None:
                img_y = scipy.ndimage.rotate(img_y, self.modes[index], reshape=False, prefilter=False, mode='reflect')

            #cv.imshow('image_out', img_x)
            #cv.waitKey(0)
            #cv.imshow('image_out', img_y)
            #cv.waitKey(0)
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

            cv.imshow('image', img_x)
            cv.waitKey(0)
            return img_x, img_y

    class ToRGB(Base):

        def __init__(self):
            super().__init__()

        def function(self, img_x, img_y):
            if img_x is not None:
                img_x = np.resize(img_x, (img_x.shape[0], img_x.shape[1], 3))
            if img_y is not None:
                img_y = np.resize(img_y, (img_y.shape[0], img_y.shape[1], 3))
            return img_x, img_y

    class Central_Crop(Base):

        def __init__(self, size=(512,512)):
            self.crop_size = size
            super().__init__()

        def function(self, img_x, img_y):

            if img_x is not None:
                y, x = img_x.shape

                if x < self.crop_size[0] and y < self.crop_size[1]:
                    raise Exception("File size to small!")

                startx = x // 2 - (self.crop_size[0] // 2)
                starty = y // 2 - (self.crop_size[1] // 2)
                img_x = img_x[starty:starty + self.crop_size[1], startx:startx + self.crop_size[0]]
            if img_y is not None:
                y, x = img_y.shape

                if x < self.crop_size[0] and y < self.crop_size[1]:
                    raise Exception("File size to small!")

                startx = x // 2 - (self.crop_size[0] // 2)
                starty = y // 2 - (self.crop_size[1] // 2)
                img_y = img_y[starty:starty + self.crop_size[1], startx:startx + self.crop_size[0]]

            return img_x, img_y

    class Crop_by_Center(Base):

        def __init__(self, treshold=25, size=(256,256)):
            self.crop_size = size
            self.treshold = treshold
            super().__init__()

        def function(self, img_x, img_y):

            if img_x is not None:
                _, mask = cv.threshold(img_x, np.max(img_x) - self.treshold, 255, cv.THRESH_BINARY)
                center_of_mass = regionprops(mask, img_x)[0].centroid
                startx = int(center_of_mass[1]) - (self.crop_size[1] // 2)
                starty = int(center_of_mass[0]) - (self.crop_size[0] // 2)

                if startx < 0:
                    startx = 0
                if starty < 0:
                    starty = 0

                if startx >= img_x.shape[1] - self.crop_size[1]:
                    startx = img_x.shape[1] - self.crop_size[1]
                if starty >= img_x.shape[1] - self.crop_size[1]:
                    starty = img_x.shape[1] - self.crop_size[1]

                img_x = img_x[starty:starty + self.crop_size[1], startx:startx + self.crop_size[0]]

                if img_y is not None:
                    img_y = img_y[starty:starty + self.crop_size[1], startx:startx + self.crop_size[0]]

            return img_x, img_y
