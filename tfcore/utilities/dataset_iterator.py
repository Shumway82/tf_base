import numpy as np
from abc import abstractmethod, ABC
from tfcore.utilities.cache import Cache
from keras.utils import Sequence, OrderedEnqueuer


class DatasetSequence(Sequence, ABC):
    """ Allow for easy iteration over batches in a dataset.
    This is a metaclass. You need to create a subclass and implement
    at least `def get_element(self, idx)` which get an element (x and y) of
    your dataset. You can also overwrite the `__init__` but don't forget to
    call `super()`.

    Example
    ```python
    class TestSequence(DatasetSequence):
        def __init__(self, batch_size, shuffle):
            self.files_x = sorted(glob.glob(TEST_DIRECTORY + "x*.jpg"))
            self.files_y = sorted(glob.glob(TEST_DIRECTORY + "y*.jpg"))
            super().__init__(len(self.files_x), batch_size, shuffle, cache_size=1)

        def get_element(self, idx):
            # Returns one sample of the dataset (not one batch!)
            # The class takes care of batching the data for you.
            img_x = imageio.imread(self.files_x[idx])
            img_y = imageio.imread(self.files_y[idx])
            return img_x, img_y

    my_sequence = TestSequence(batch_size=32, shuffle=True)

    # You can use it as an iterator.
    for i in range(nb_epochs):
        for batch_x, batch_y in my_sequence:
            # Train on the batches here.

    # You can also use it with indexing.
    for i in range(nb_epochs):
        for j in range(len(my_sequence)):
            batch_x, batch_y = my_sequence[j]
            # Train on the batches here.
    ```

    # Arguments
        nb_samples: The number of samples (not batches) in the dataset.
        batch_size: The size of the batches to use.
        shuffle: Set to True if you want the class to take the samples in a random order.
        cache_size: The size of the cache to use. Default is 0 (so no caching).
        cache_function: By default, if cache_size is not 0, the iterator is going to
            cache the results of the `self.get_element(self, idx)` function.
            But if you rather want a cache that you can manually use, you can give the cache
            function to execute, the cache will use this rather than `self.get_element(self, idx)`.
            You'll then be able to access the `Cache` object with `self.user_cache` and use the
            cache function with `self.user_cache.load(...)`. For more info on that, see
            the `Cache` class in the repo video_utilities.
    """

    def __init__(self, nb_samples, batch_size, shuffle=True, cache_size=0, cache_function=None):
        if nb_samples <= 0:
            raise ValueError("The size of your dataset isn't valid: " + str(nb_samples))
        self.nb_samples = nb_samples
        self.batch_size = batch_size

        if cache_function is None:
            self.user_cache = None
            self._cache_element_dataset = Cache(cache_size, nb_samples, self.get_element)
        else:
            self.user_cache = Cache(cache_size, nb_samples, cache_function)
            self._cache_element_dataset = Cache(0, nb_samples, self.get_element)

        self.indexes = np.arange(0, self.nb_samples, dtype=np.uint32)
        if shuffle:
            np.random.shuffle(self.indexes)
        self._internal_index = 0

        self.nb_batches = int(np.ceil(self.nb_samples / self.batch_size))

    def __len__(self):
        return self.nb_batches

    def __getitem__(self, idx):
        assert idx < self.nb_batches
        start = idx * self.batch_size
        end = min(start + self.batch_size, self.nb_samples)

        data_batch = [self._cache_element_dataset.load(i) for i in self.indexes[start:end]]

        len_tup = len(data_batch[0])
        if len_tup != 2:
            ValueError("The function get_element() should return a tuple (x, y)"
                       "but we received an object of length " + str(len_tup))

        batch = []
        for i in range(2):
            network_inputs = [x[i] for x in data_batch]
            batch.append(_concatenate_inputs(network_inputs))
        return batch

    def __iter__(self):
        return self

    def __next__(self):
        if self._internal_index < self.nb_batches:
            self._internal_index += 1
            return self[self._internal_index - 1]
        else:
            self._internal_index = 0
            raise StopIteration

    @abstractmethod
    def get_element(self, idx):
        raise NotImplementedError


class ParallelWrapper:
    """ Allow to use the class `DatasetSequence` with multithreading or multiprocessing.

    How to use:
    ```python
    class TestSequence(DatasetSequence):
        def __init__(self, batch_size, shuffle):
            self.files_x = sorted(glob.glob(TEST_DIRECTORY + "x*.jpg"))
            self.files_y = sorted(glob.glob(TEST_DIRECTORY + "y*.jpg"))
            super().__init__(len(self.files_x), batch_size, shuffle, cache_size=1)

        def get_element(self, idx):
            # Returns one sample of the dataset (not one batch!)
            # The class takes care of batching the data for you.
            img_x = imageio.imread(self.files_x[idx])
            img_y = imageio.imread(self.files_y[idx])
            return img_x, img_y

    my_sequence = TestSequence(batch_size=32, shuffle=True)
    my_sequence_multithreading = ParallelWrapper(my_sequence)

    # You can only use it as an iterator.
    for i in range(nb_epochs):
        for batch_x, batch_y in my_sequence_multithreading:
            # Train on the batches here.


    # By setting more than one worker, it automatically use multiprocessing.
    my_sequence_multiprocessing = ParallelWrapper(my_sequence, workers=3)

    for i in range(nb_epochs):
        for batch_x, batch_y in my_sequence_multiprocessing:
            # Train on the batches here.

    ```

    # Arguments
        sequence: The DatasetSequence object to wrap.
        max_queue_size: Internally, multithreading and multiprocessing are using queues.
            This sets the size of this queue.
        workers: If set to 1, uses multithreading. If set to more than one, uses
            multiprocessing. In this case it's the number of processes to spawn.

    To stop the processes or threads, you can wait until the object
    is garbage collected, or you can use `del`.
    """

    def __init__(self, sequence, max_queue_size=10, workers=1):
        self.enqueuer = OrderedEnqueuer(sequence, use_multiprocessing=workers >= 2)
        self.enqueuer.start(workers, max_queue_size)
        self.generator = self.enqueuer.get()
        self._internal_index = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self._internal_index < len(self):
            self._internal_index += 1
            return next(self.generator)
        else:
            self._internal_index = 0
            raise StopIteration

    def __len__(self):
        return len(self.enqueuer.sequence)

    def __del__(self):
        self.enqueuer.stop()


def _concatenate_inputs(list_of_lists):
    if isinstance(list_of_lists[0], np.ndarray):
        # Special case, we just concatenate.
        return np.array(list_of_lists)
    list_of_batched_input = []
    for i in range(len(list_of_lists[0])):
        tmp_list = []
        for sample in list_of_lists:
            tmp_list.append(sample[i])
        list_of_batched_input.append(np.array(tmp_list))

    return list_of_batched_input
