import abc
import threading

from tfcore.utilities.utils import *
from tfcore.utilities.params_serializer import ParamsSerializer


class ITrainer_Params(ParamsSerializer):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def __init__(self):
        super().__init__()
        self.is_train = True
        self.gpu = True
        self.gpus = [0]  # [0, 1, 2, 3]
        self.batch_size = 16
        self.epoch = 1
        self.new = True
        self.checkpoint_dir = 'checkpoints'
        self.make_summery_full = False
        self.make_summery_graph = False
        self.use_tensorboard = False


class ITrainer():
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def __init__(self, params):
        self.params = params

        print('Libs loaded')
        print('Tensorflow ' + tf.__version__)
        gpus = self.params.gpus  # Here I set CUDA to only see one GPU
        os.environ["CUDA_VISIBLE_DEVICES"] = ','.join([str(i) for i in gpus])

        if self.params.gpu:
            gpu_nums = len(self.params.gpus)
        else:
            gpu_nums = 0

        gpu_options = tf.GPUOptions()
        gpu_options.allocator_type = 'BFC'
        gpu_options.allow_growth = True

        config = tf.ConfigProto(allow_soft_placement=True,
                                log_device_placement=False,
                                gpu_options=gpu_options,
                                device_count={'GPU': gpu_nums})
        #config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1

        self.sess = tf.Session(config=config)

        self.summaries_val = []
        self.summaries_vis = []
        self.summaries_vis_one = []
        self.summaries = []

        self.summary_val = None
        self.summary_vis = None
        self.summary_vis_one = None
        self.summary = None

        self.model_dir = 'model'
        self.sample_dir = 'samples'
        self.checkpoint_dir = 'checkpoint'
        self.log_dir = 'logs'

        self.saver = None
        self.writer = None
        self.models = []

        self.init_global_step = 0
        if not self.params.new:
            self.init_global_step = get_global_steps(self.params.checkpoint_dir)
        self.epoch = tf.placeholder(tf.int32, name='epoch')
        self.global_step = tf.Variable(self.init_global_step, trainable=False)

        self.batch_size_total = self.params.batch_size
        self.batch_size = int(self.batch_size_total / len(self.params.gpus))

    @abc.abstractmethod
    def __del__(self):
        tf.reset_default_graph()
        self.sess.close()
        print('[*] Session closed...')

    def launch_tensorboard(self):
        import os
        os.system('tensorboard --logdir=' + self.log_dir + ' --port=8009')
        return

    def make_summarys(self, gradient_list):

        if self.params.make_summery_full:
            for var in tf.trainable_variables():
                self.summaries_val.append(tf.summary.histogram(var.op.name, var))

            for grads in gradient_list:
                for grad, var in grads:
                    self.summaries_val.append(tf.summary.histogram(var.op.name + '/gradients',
                                                                   grad))
            print(' [*] Full Summery created...')

        for model in self.models:
            self.summaries_val.extend(model.summary_val)
            self.summaries_vis.extend(model.summary_vis)
            self.summaries_vis_one.extend(model.summary_vis_one)
            self.summaries.extend(model.summary)

        self.summary_val = tf.summary.merge([self.summaries_val])
        self.summary_vis = tf.summary.merge([self.summaries_vis])
        self.summary_vis_one = tf.summary.merge([self.summaries_vis_one])
        self.summary = tf.summary.merge([self.summaries])

        if self.params.make_summery_graph:
            self.writer = tf.summary.FileWriter(os.path.join(self.log_dir), self.sess.graph)
        else:
            self.writer = tf.summary.FileWriter(os.path.join(self.log_dir))

        if self.params.use_tensorboard:
            t = threading.Thread(target=self.launch_tensorboard, args=([]))
            t.start()
        print(' [*] Log-file ' + os.path.join(self.log_dir) + ' created...')

    @abc.abstractmethod
    def validate(self, epoch, counter, idx):
        raise NotImplementedError("Please Implement this method")

    @abc.abstractmethod
    def build_model(self, tower_id):
        raise NotImplementedError("Please Implement this method")

    @staticmethod
    def compute_tower_gradients(model, max_norm=5.0):
        model.vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=model.params.scope)
        grads = model.optimizer.compute_gradients(model.total_loss, var_list=model.vars)
        grads = [(tf.clip_by_norm(grad, max_norm), var) for grad, var in grads]
        print(' [*] Gradients of ' + model.params.name + ' computed...')
        return grads

    def build_pipeline(self):

        if self.params.gpu:
            device = "/gpu:"
            #device = "/job:localhost/replica:0/task:0/device:XLA_GPU:"
        else:
            device = "/cpu:"
        tower_index = 0
        with tf.variable_scope(tf.get_variable_scope()):
            for tower_id in self.params.gpus:
                with tf.device(device + '%d' % tower_id):
                    with tf.name_scope('Tower_%d' % (tower_index)) as scope:
                        print(' [*] Tower ' + str(tower_index) + ' ' + device + str(tower_id))

                        self.models = self.build_model(tower_id=tower_index)

                        tf.get_variable_scope().reuse_variables()
                        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                        with tf.control_dependencies(update_ops):
                            for i in range(len(self.models)):
                                gradients = self.compute_tower_gradients(self.models[i])
                                self.models[i].gradients.append(gradients)
                        tower_index += 1
        gradient_list = []

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            for i in range(len(self.models)):
                if tower_index > 1:
                    grads = average_gradients(self.models[i].gradients)
                    print(' [*] Gradients of ' + self.models[i].params.name + ' averaged...')
                else:
                    grads = self.models[i].gradients[0]
                self.models[i].optimizer = self.models[i].optimizer.apply_gradients(grads,
                                                                                    global_step=self.global_step)
                gradient_list.append(grads)
                print(' [*] Gradients of ' + self.models[i].params.name + ' applyed...')

        self.make_summarys(gradient_list)
        print(' [*] Build pipeline pass...')
        return

    @abc.abstractmethod
    def train_online(self, batch_X, batch_Y, epoch=0, counter=1, idx=0, batch_total=0):
        raise NotImplementedError("Please Implement this method")

    @abc.abstractmethod
    def train(self):
        raise NotImplementedError("Please Implement this method")
