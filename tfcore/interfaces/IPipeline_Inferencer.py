"""
Copyright (C) Silvio Jurk - All Rights Reserved
Unauthorized copying of this file, via any medium is strictly prohibited
Proprietary and confidential.

@author: Silvio Jurk
"""

import imageio

from tfcore.interfaces.IInferencing import *
from tfcore.utilities.dataset_iterator import *
from tfcore.utilities.params_serializer import ParamsSerializer
from tfcore.utilities.files import get_img_paths

error_name_string = ('The function name {} appears '
                     'at least two times in the {} pipeline. '
                     'please use the argument \'name\' of this function to '
                     'solve this issue.')

s = ('input_X', 'output_Y')


class IPipeline_Inferencer_Params(ParamsSerializer):
    """ Parameter class  for IPipeline

            # Arguments

                data_dir: Directory of the ground truth training data Y.
                convert: Create input data set X from ground truth Y.
                resize_factor: Scaling factor for input X
                interp: Interpolation technique for converting input X from ground truth Y
    """

    def __init__(self,
                 data_dir,
                 data_dir_x='',
                 convert=True,
                 resize_factor=2,
                 interp='bicubic'):

        self.data_dir = data_dir
        self.data_dir_x = data_dir_x
        self.resize_factor = resize_factor
        self.convert = convert
        self.interp = interp

        super().__init__()


class IPipeline_Inferencer(DatasetSequence):
    """ Allow the easy inferencing of generator models.
    This is an metaclass. You need to create a subclass and you can override
    the `def get_element(self, idx)` which get an element (x and y) of
    your dataset. You can also overwrite the `__init__` but don't forget to
    call `super()`.

    Example
    ```python
    class My_Pipeline(IPipeline):
        def __init__(self, trainer, params, pre_processing):
            # Init super class IPipeline
            super().__init__(trainer, params, pre_processing)

        def get_element(self, idx):
            # Returns one sample of the dataset (not one batch!)
            # The class takes care of batching the data for you.
            img_x = imageio.imread(self.files_y[idx])
            # Process data augmentation, scaling and coding with h264
            img_x, img_y = self.pre_processing.run(img_x, img_x)
            return img_x, img_x
    ```

    # Arguments
        inferencer: Implementation of meta class IInferencer
        params: Implementation of meta class IPipeline_Params
        pre_processing: Implementation of class Preprocessing

    """
    def __init__(self, inferencer, params, pre_processing=None):

        if not isinstance(inferencer, IInferencing):
            raise KeyError(' [!] Parameter trainer is not of type ITrainer')

        if not isinstance(params, IPipeline_Inferencer_Params):
            raise KeyError(' [!] Parameter params is not of type Pipeline_Params')

        self.inferencer = inferencer
        self.params = params
        self.pre_processing = pre_processing

        self.files_y = get_img_paths(self.params.data_dir)
        if len(self.files_y) == 0:
            raise FileNotFoundError(' [!] No files in data-set')

        if self.params.data_dir_x != '':
            self.files_x = get_img_paths(self.params.data_dir_x)
            if len(self.files_x) == 0:
                raise FileNotFoundError(' [!] No files in data-set')
            self.params.convert = False


        super().__init__(len(self.files_y),
                         1,
                         False)

    def get_element(self, idx):

        if self.params.convert:
            try:
                img_y = imageio.imread(self.files_y[idx])
                img_x = resize(img_y, (int(img_y.shape[0] / self.params.resize_factor), int(img_y.shape[1] / self.params.resize_factor)))
            except FileNotFoundError:
                raise FileNotFoundError(' [!] File not found of data-set Y')
        else:
            try:
                img_x = imageio.imread(self.files_y[idx])
            except FileNotFoundError:
                raise FileNotFoundError(' [!] File not found of data-set X')

        if self.pre_processing is not None:
            img_x, _ = self.pre_processing.run(img_x, None)

        return img_x, img_x

    def run(self):
        try:
            batch_x, batch_y = self.__next__()
            img_out_x= self.inferencer.inference(batch_x[0])
        except StopIteration as err:
                print(' [*] Inferencing Finished..', err)
                return None
        except Exception as err:
                print(' [!] Error in Inferencer in inference():', err)
                raise
        return img_out_x[0]
