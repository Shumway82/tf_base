"""
Copyright (C) Silvio Jurk - All Rights Reserved
Unauthorized copying of this file, via any medium is strictly prohibited
Proprietary and confidential.

@author: Silvio Jurk
"""

import abc
from tfcore.core.loss import *
from tfcore.utilities.utils import *
from tfcore.utilities.params_serializer import ParamsSerializer


class IModel_Params(ParamsSerializer):
    __metaclass__ = abc.ABCMeta

    def __init__(self,
                 learning_rate=0.0001,
                 lr_lower_bound=1e-5,
                 beta1=0.9,
                 global_steps=0,
                 steps_decay=4000,
                 decay=1.0,
                 scope='scope',
                 name='model'):
        super().__init__()

        self.scope = scope
        self.name = name
        self.learning_rate = learning_rate
        self.lr_lower_bound = lr_lower_bound
        self.beta1 = beta1
        self.global_steps = global_steps
        self.step_decay = steps_decay
        self.decay = decay
        self.max_images = 6
        self.path = os.path.realpath(__file__)

    def load(self, path):
        return super().load(os.path.join(path, self.name))

    def save(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
        super().save(os.path.join(path, self.name))
        return


class IModel():
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def __init__(self, sess, params, global_steps):
        self.params = params
        self.sess = sess
        self.summary_vis = []
        self.summary_vis_one = []
        self.summary_val = []
        self.summary = []
        self.input_stage = ''
        self.output_stage = ''
        self.max_images = 6
        self.total_loss = 0
        self.gradients = []
        self.global_steps = global_steps
        self.set_optimizer()
        self.G = None
        self.crl = CLR(sess,
                       start_lr=self.params.lr_lower_bound,
                       end_lr=self.params.learning_rate,
                       step_size=self.params.step_decay,
                       gamma=self.params.decay)
        print('Model: ' + params.name)

    def build_model(self, input, is_train=False, reuse=False):
        """Retrieve data from the input source and return an object."""
        self.params.input_stage = input.name

        self.G = self.model(input, is_train=is_train, reuse=reuse)

        self.params.output_stage = self.G.name.split(':')[0]

        print(' [*] Input-Stage: ' + self.params.input_stage)
        print(' [*] Output-Stage: ' + self.params.output_stage)
        print(' [*] Model ' + self.params.name + ' initialized...')
        return self.G

    def set_optimizer(self):
        self.learning_rate = tf.placeholder(tf.float32, shape=[])
        with tf.variable_scope(self.params.scope):
            self.optimizer = tf.train.AdamOptimizer(self.learning_rate,
                                                    beta1=self.params.beta1,
                                                    name='Adam')
            self.summary.append(tf.summary.scalar("lr_" + self.params.scope, self.learning_rate))

    @abc.abstractmethod
    def model(self, input, is_train=False, reuse=False):
        return input

    def loss(self, Y, normalize=False, name='MSE'):
        """Retrieve data from the input source and return an object."""
        loss_function = get_loss_function(name)
        if normalize:
            return loss_normalization(loss_function(Y, self.G))
        else:
            return loss_function(Y, self.G)

    def inference(self, input):
        """Retrieve data from the input source and return an object."""
        return self.G.eval({self.inputs: input})

    def load(self, path):
        return load_model(self.sess, path, model_name=self.params.name, scope=self.params.scope)

    def save(self, path, global_step=0):
        return super().save(self.sess, path, model_name=self.params.name, scope=self.params.scope,
                            global_step=global_step)

    def save(self, sess, path, global_step=0):
        model_dir = save_model(sess, path, model_name=self.params.name, scope=self.params.scope,
                               global_step=global_step)
        try:
            self.params.save(model_dir)
            shutil.copy2(self.params.path, model_dir)
        except Exception as err:
            print(' [!] Can not copy model file ' + self.params.path + ' to ' + model_dir + ', ' +
                  str(err))

    def get_summary(self):
        return self.get_summary

    def get_summary_val(self):
        return self.get_summary_val

    def get_summary_vis(self):
        return self.get_summary_vis
