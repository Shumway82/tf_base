"""
Copyright (C) Silvio Jurk - All Rights Reserved
Unauthorized copying of this file, via any medium is strictly prohibited
Proprietary and confidential.

@author: Silvio Jurk
"""

from tfcore.interfaces.ITraining import *
from tfcore.utilities.params_serializer import ParamsSerializer

print('Libs loaded')
print(tf.__version__)
gpus = [0, 1, 2, 3]  # Here I set CUDA to only see one GPU
os.environ["CUDA_VISIBLE_DEVICES"] = ','.join([str(i) for i in gpus])
num_gpus = len(gpus)  # number of GPUs to use


class IInferencer_Params(ParamsSerializer):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def __init__(self):
        super().__init__()
        self.is_train = False
        self.gpu = True
        self.gpus = [0]
        self.batch_size = 1


class IInferencing():
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def __init__(self, params):
        self.params = params
        if self.params.gpu:
            gpu_nums = 1
        else:
            gpu_nums = 0

        gpu_options = tf.GPUOptions()
        gpu_options.allocator_type = 'BFC'
        gpu_options.allow_growth = True
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                     log_device_placement=False,
                                                     gpu_options=gpu_options,
                                                     device_count={'GPU': gpu_nums}))

    @abc.abstractmethod
    def __del__(self):
        tf.reset_default_graph()
        self.sess.close()
        print('[*] Session closed...')

    @abc.abstractmethod
    def inference(self, input):
        raise NotImplementedError("Please Implement this method")
