from random import randint
import numpy as np
from scipy.misc import imresize


def random_crop(array, crop_shape, ignore_size_errors=False):
    """ Take a crop at a random position in an image.
    I works also for batches of images.
    For batches, the position of the crop is always the same.

    There are three ways to call this function:

    ```python
    >>> image = np.zeros((720, 1280, 3))
    >>> random_crop(image, (200, 100)).shape
    (200, 100, 3)

    >>> batch = np.zeros((32, 720, 1280, 3))
    >>> random_crop(batch, (200, 100)).shape
    (32, 200, 100, 3)

    >>> batch = np.zeros((32, 720, 1280, 3))
    >>> random_crop(batch, (4, 200, 100, 3)).shape
    (4, 200, 100, 3)
    ```

    # Arguments
        array: A numpy array to crop. Can be 3D or 4D.
        crop_shape: One integer or tuple of two/three integers.
            If a single integer is given, it represents the size of a
            square crop.
            Three integers is if you want to crop the batch axis too.
        ignore_size_errors: Don't throw an error if the image is too
            small for the crop. Returns the full image.
    """

    if isinstance(crop_shape, int):
        crop_shape = (crop_shape, crop_shape)
    crop_shape = list(crop_shape)

    if array.ndim == 4 and len(crop_shape) == 2:
        # We take all the images in the batch
        crop_shape = [array.shape[0]] + crop_shape

    if ignore_size_errors:
        crop_shape = [min(x, y) for x, y in zip(array.shape, crop_shape)]

    starts = [randint(0, x - y) for x, y in zip(array.shape, crop_shape)]

    slicing = tuple(np.s_[x: x + y] for x, y in zip(starts, crop_shape))

    return array[slicing]


def center_crop_resize(array, new_shape):
    if new_shape is None:
        return array
    assert new_shape[0] == new_shape[1]
    if array.shape[0] > array.shape[1]:
        start = (array.shape[0] // 2) - (array.shape[1] // 2)
        array = array[start: start + array.shape[1]]
    elif array.shape[0] < array.shape[1]:
        start = (array.shape[1] // 2) - (array.shape[0] // 2)
        array = array[:, start: start + array.shape[0]]
    return resize(array, new_shape)


def resize(array, new_shape, mode='bilinear'):
    """ resize the image with different modes.
    It works also for batches of images.
    how to import : `from utilities.image import resize`

    # Arguments
        array: A numpy array to crop. Can be 3D or 4D.
        new_shape: tuple of two/three integers.
        mode: he mode can be either `top_left`, `random_crop`,
        `cnn`, `nearest`, `lanczos`, `bilinear`, `bicubic` or `cubic`.
        *** `cnn`: for mode='cnn': see the function: center_crop_resize`, it
        crops the center of image, you can resize this crop.
        usage: neural network

    """

    if array.shape[:2] != new_shape:
        if mode == 'top_left':
            return array[:new_shape[0], :new_shape[1]]
        elif mode == 'random_crop':
            return random_crop(array, new_shape)
        elif mode == 'cnn':
            return center_crop_resize(array, new_shape)
        else:
            if array.ndim == 4:
                output_list = [imresize(img, new_shape, mode) for img in array]
                return np.stack(output_list, axis=0)
            else:
                return imresize(array, new_shape, mode)
    return array
