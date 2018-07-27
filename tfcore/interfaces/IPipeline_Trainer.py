"""
Copyright (C) Silvio Jurk - All Rights Reserved
Unauthorized copying of this file, via any medium is strictly prohibited
Proprietary and confidential.

@author: Silvio Jurk
"""

import imageio

from tfcore.interfaces.ITraining import *
from tfcore.utilities.dataset_iterator import *
from tfcore.utilities.params_serializer import ParamsSerializer
from tfcore.utilities.files import get_img_paths

error_name_string = ('The function name {} appears '
                     'at least two times in the {} pipeline. '
                     'please use the argument \'name\' of this function to '
                     'solve this issue.')

s = ('input_X', 'output_Y')


class IPipeline_Trainer_Params(ParamsSerializer):
    """ Parameter class  for IPipeline

    # Arguments

        data_dir: Directory of the ground truth training data Y.
        validation_dir: Directory of the validation data set.
        output_dir: Output directory for logs, samples and checkpoints.
        data_dir_x: Directory of input data set X. If it is None it will
        be created from ground truth Y by scaling.
        convert: Create input data set X from ground truth Y.
        resize_factor: Scaling factor for input X
        epochs: Epochs for iterating over the same data set
        batch_size: Numbers of images in a training batch
        shuffle: Set to True if you want the class to take the samples in a random order.
        cache_size: The size of the cache to use. Default is 0 (so no caching).
        use_high_space: If True, input image have the same size like output image
        interp: Interpolation technique for converting input X from ground truth Y
    """

    def __init__(self,
                 data_dir,
                 validation_dir,
                 output_dir,
                 data_dir_x=None,
                 convert=True,
                 resize_factor=2,
                 epochs=25,
                 batch_size=16,
                 shuffle=True,
                 cache_size=1,
                 use_high_space=False,
                 interp='bicubic'):

        self.data_dir = data_dir
        self.data_dir_x = data_dir_x
        self.validation_dir = validation_dir
        self.output_dir = output_dir
        self.resize_factor = resize_factor
        self.epochs = epochs
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.cache_size = cache_size
        self.convert = convert
        self.interp = interp
        self.use_high_space = use_high_space

        super().__init__()


class IPipeline_Trainer(DatasetSequence):
    """ Allow the easy training and evaluation of generator models.
    This is an metaclass. You need to create a subclass and you can override
    the `def get_element(self, idx)` which get an element (x and y) of
    your dataset. You need to implement the method set_validation(self), witch
    set the validation set for the trainer. You can also overwrite the `__init__`
    but don't forget to call `super()`.

    Example
    ```python
    class My_Pipeline(IPipeline):
        def __init__(self, trainer, params, pre_processing):
            # Init super class IPipeline
            super().__init__(trainer, params, pre_processing)

        def set_validation(self):
            # Load files from validation directory using utilities-class
            files_val_y = get_img_paths(self.params.validation_dir)
            batch_val_y = np.asarray([imageio.imread(file) for file in files_val_y])
            # Prepare input X by scaling of ground truth Y
            batch_val_x = resize(batch_val_y, self.img_shape_x, self.params.interp)
            # Set batch for X and Y to trainer
            self.trainer.set_validation_set(batch_val_x, batch_val_y)

        def get_element(self, idx):
            # Returns one sample of the dataset (not one batch!)
            # The class takes care of batching the data for you.
            img_x = imageio.imread(self.files_x[idx])
            img_y = imageio.imread(self.files_y[idx])
            # Process data augmentation, scaling and coding with h264
            img_x, img_y = self.pre_processing.run(img_x, img_y)
            return img_x, img_y
    ```

    # Arguments
        trainer: Implementation of meta class ITrainer
        params: Implementation of meta class IPipeline_Params
        pre_processing: Implementation of class Preprocessing

    """
    def __init__(self, trainer, params, pre_processing=None):

        if not isinstance(trainer, ITrainer):
            raise KeyError(' [!] Parameter trainer is not of type ITrainer')

        if not isinstance(params, IPipeline_Trainer_Params):
            raise KeyError(' [!] Parameter params is not of type Pipeline_Params')

        self.trainer = trainer
        self.params = params
        self.pre_processing = pre_processing

        self.user_functions_code = ([], [])

        self.files_y = get_img_paths(self.params.data_dir)
        if len(self.files_y) == 0:
            raise FileNotFoundError(' [!] No files in data-set')

        self.img_shape_y = imageio.imread(self.files_y[0]).shape

        if self.params.data_dir_x is not None:
            self.files_x = get_img_paths(self.params.data_dir_x)
            if len(self.files_x) == 0:
                self.params.convert = True
                print(' [!] No files for X found. Switch to convert from Y')
        else:
            self.params.convert = True

        if self.params.convert:
            self.img_shape_x = [int(self.img_shape_y[0] / self.params.resize_factor),
                                int(self.img_shape_y[1] / self.params.resize_factor),
                                3]
        else:
            self.img_shape_x = imageio.imread(self.files_x[0]).shape

        super().__init__(len(self.files_y),
                         self.params.batch_size,
                         self.params.shuffle,
                         cache_size=self.params.cache_size)

    @abc.abstractmethod
    def set_validation(self, resize_factors_hs, use_hs=False):
        raise NotImplementedError("Please Implement this method")

    def get_element(self, idx):

        try:
            img_y = imageio.imread(self.files_y[idx])
        except FileNotFoundError:
            raise FileNotFoundError(' [!] File not found of data-set Y')

        if not self.params.convert:
            try:
                img_x = imageio.imread(self.files_x[idx])
            except FileNotFoundError:
                raise FileNotFoundError(' [!] File not found of data-set X')
        else:
            img_x = img_y

        if self.pre_processing is not None:
            img_x, img_y = self.pre_processing.run(img_x, img_y)

        return img_x, img_y

    def run(self):

        counter = 0
        for epoch in range(self.params.epochs):
            idx = 0
            for batch_x, batch_y in self:
                try:
                    self.trainer.train_online(batch_X=batch_x,
                                              batch_Y=batch_y,
                                              epoch=epoch,
                                              counter=counter,
                                              idx=idx,
                                              batch_total=self.nb_batches)
                except Exception as err:
                    print(' [!] Error in Trainer in train_online():', err)
                    raise

                counter += 1
                idx += 1
