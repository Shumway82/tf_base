try:
    import tensorflow as tf
    from keras.models import load_model
except ImportError as e:
    print("Tensorflow-gpu have to be installed! : ", e)

from tfcore.interfaces.IInferencing import *
import time
import abc



class FileType(object):
    """This is an enum and lists all supported kinds of how a model can be loaded for this project.
            # Model Types
                Tensorflow : .meta, .index, .data-..
                Keras : .h5, .hdf5
                Protobuf : .pb
                Uff : .uff
    """
    Tensorflow = 1
    Keras = 2
    Protobuf = 3
    Uff = 4


class FrozenInferencerParams(IInferencer_Params):
    """This is the container class for our class Inferencer for storing all relevant varibales.
    Additionally, with this class it is possible to load and save all public class-variables into
    a human readable textfile by using json
    """

    def __init__(self, model_path=None, file_type=FileType.Protobuf):
        """Constructor for a class that contains all relevant parameters for the general Inferencer
        (without TensorRT)

        # Parameters
             model_path : as a string of the path of the model file (for tensorflow it's the e.g.
                ".index"-file). For using tensorflow, call the .index-file path. Generally, call
                the models without the file-suffix!
             file_type : a flag to say how the model shall be loaded by the model_path. By default
                a tensorflow model try to be loaded.
             channels : number of color channels as int
             height : height of the input image
             width : width of the input image
             input_node : a string of the input that is defined by the model
             output_node : a string of the output that is defined by the model
        """
        self.channels = 3
        if model_path is None:
            base_path = os.path.dirname(os.path.realpath(__file__))
            self.model_path = base_path + "/models/DRRN_B1U7_R2/ExampleModel-2194"
        else:
            self.model_path = model_path
        self.file_type = file_type
        self.input_node = "inputs"
        self.output_node = "AddN"
        self.is_train = False
        self.gpu = True
        self.gpus = [0]
        self.batch_size = 1


class FrozenInferencer(IInferencing):
    __metaclass__ = abc.ABCMeta
    """This class serves as an abstract base class for implementing a specialized deep learning
    inferencer. This class provides a basic inferencing functionality running your network on a CPU
    or GPU.
    """

    def __init__(self, params):
        """
        Constructor for setting all relevant parameters. The model for inferencing will be set in a
        hierarchical order. A given session will be used by all means. In any other case the model
        will be loaded by the model_path. Note, that you have to say what type of model are included
        by the given path (set )

        # Arguments
             session : the whole session itself that is currently living by a framework.
             dtype : what data type the model (means nodes, input and output) has
        """

        if not type(params) is FrozenInferencerParams:
            print('ERROR : Argument "params" has to be an instance of class "InferencerParams"!')
            return

        if params.model_path is "":
            print('No valid session or model_path was given! session is None and params.model_path '
                  'is ""')
            return

        self.model_path = params.model_path
        self.input_node = params.input_node
        self.output_node = params.output_node
        self.file_type = params.file_type

        if self.file_type is FileType.Tensorflow:
            self.sess = self._get_tensorflow_sess(params.model_path)
        elif self.file_type is FileType.Keras:
            self.sess = self._get_keras_sess(params.model_path)
        elif self.file_type is FileType.Protobuf:
            self.sess = self._get_protobuf_sess(params.model_path)
        elif self.file_type is FileType.Uff:
            # nothing to do here. it's part of class TensorRTInferencer
            return
        else:
            super().__init__(params)

        if self.file_type is not FileType.Protobuf:
            self.frozen_graphdef = self.get_frozen_model(sess=self.sess,
                                                         output_node=self.output_node)

            tf.import_graph_def(self.frozen_graphdef)
            graph = tf.get_default_graph()

            self._set_io_by_graph(graph)

    def __del__(self):
        try:
            print("[*] TensorRT engine disposed")
        except ValueError:
            print("[-] TensorRT engine interrupted")

    @staticmethod
    def _get_tensorflow_sess(model_path):
        """Get a Tensorflow session by a given model path for Tensorflow models
        """
        tf.reset_default_graph()

        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
        session = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        imported_meta = tf.train.import_meta_graph(model_path + ".meta")
        imported_meta.restore(session, model_path)

        return session

    @staticmethod
    def _get_keras_sess(model_path):
        """Get a Tensorflow session by a given model path for keras models
        """
        model = load_model(model_path, compile=False)
        # Get model input and output names
        model_input = model.input.name.strip(':0')
        model_output = model.output.name.strip(':0')

        print("Input/Output : ", model_input, model_output)

        session = tf.keras.backend.get_session()

        return session

    def _get_protobuf_sess(self, model_file):
        """This metthod will be called by a given protobuf file .pb . A protobuf file can be created
        by serializing a tensorflow graph.
        """
        tf.reset_default_graph()

        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)

        with tf.gfile.GFile(model_file + '.pb', "rb") as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())

        with tf.Graph().as_default() as graph:
            tf.import_graph_def(graph_def, name="")

        self._set_io_by_graph(graph)

        session = tf.Session(graph=graph, config=tf.ConfigProto(gpu_options=gpu_options))

        return session

    @staticmethod
    def get_frozen_model(sess, output_node):
        """Create and gets an optimized frozen graph from a given tensorflow session.
        Means, that all variables will be changed into constants and training nodes will be removed.
        This is commonly used for speed up the inferencing process. Additionally, this method
        saves the frozen model (protobuf-model) to a file to the given model_path-location.
        """
        graphdef = tf.get_default_graph().as_graph_def()
        frozen_graph = tf.graph_util.convert_variables_to_constants(sess, graphdef,
                                                                    [output_node])
        frozen_graph = tf.graph_util.remove_training_nodes(frozen_graph)

        return frozen_graph

    @staticmethod
    def save_as_protobuf(frozen_graphdef, model_path):
        """Saves a created tensorflow graphdef as a protobuf serialized file
                # Argument
                    model_path : as string with the file + model name. By default it's
                      model_path + tf_graph.pb
        """

        with tf.gfile.GFile(model_path + '.pb', "w") as f:
            f.write(frozen_graphdef.SerializeToString())

    def _set_io_by_graph(self, graph):
        """This method just sets the input and output tensors for our tensorflow run() method of
        our session
        """
        self.x = graph.get_tensor_by_name(self.input_node + ":0")
        self.y = graph.get_tensor_by_name(self.output_node + ":0")

    def inference(self, input):

        input = normalize(input).astype(np.float)
        input = np.rot90(input, k=1)
        w, h, c = input.shape

        sample = np.asarray(input.reshape(1, w, h, c))

        start_time = time.time()

        try:
            imageOut = self.sess.run(self.y, feed_dict={self.x: sample})
        except BaseException as ex:
            print("Exception during the inferencing process with msg: ", str(ex))
        time_ms = (time.time() - start_time) * 1000.0
        fps = 0
        try:
            fps = 1000.0 / time_ms
            print("Inferencing: time: %4.4f ms, fps: %.2f"
                  % (time_ms, fps))
            imageOut = inormalize(imageOut[0])
            imageOut = np.rot90(imageOut, k=-1)
            imageOut = np.clip(imageOut, a_min=0, a_max=255)
            return imageOut.astype(np.uint8), fps
        except BaseException as ex:
            print("Inferencing: error", str(ex))
            return None, 0
