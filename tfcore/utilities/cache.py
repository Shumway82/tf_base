"""
Copyright (C) Silvio Jurk - All Rights Reserved
Unauthorized copying of this file, via any medium is strictly prohibited
Proprietary and confidential.

@author: Silvio Jurk
"""

from random import random
from tfcore.utilities.size import compute_size


class Cache:
    """ Cache files so that you don't have to load them all the time.
            Is pretty efficient when iterating over a dataset.

            # Arguments
                max_size_GB: The maximum memory size in GigaBytes that the cache
                    can use.
                nb_of_files: The number of files in the dataset.
                    Is used to estimate the total unpacked size of
                    the dataset. So that the cache doesn't cache
                    only files at the start of the iteration.
                loading_function: function to use to load a file.
                    Must take a path as the first argument.
                    It'll be skipped if there is a cache hit.

            # Usage

            ```python
            from imageio import imread
            from utilities import Cache
            import glob

            files = glob.glob("datadir/*.jpg")
            cache = Cache(4, len(files), imread)

            for _ in range(10):
                for file in files:
                    # The jpeg decoding will be skipped if there is a cache hit
                    # speeding up the whole iteration.
                    image_as_array = cache.load(file)
                    # Do things with the numpy array...
            ```
            """

    def __init__(self, max_size_GB, nb_of_files, loading_function):
        self.max_size_GB = max_size_GB
        self.nb_of_files = nb_of_files
        self.loading_function = loading_function

        self.cached_data = {}
        self.current_cache_size = 0
        self.probability_cache = None
        self.nb_files_loaded = 0
        self.total_size_seen = 0  # The size in GB of all the files seen so far.

    def load(self, file_path):
        try:
            return self.cached_data[file_path]
        except KeyError:
            data = self.loading_function(file_path)
            if self.max_size_GB > 0:
                self._subroutine(data, file_path)
            return data

    def _subroutine(self, loaded_data, file_path):
        data_size = compute_size(loaded_data) / (2 ** 30)
        if self.nb_files_loaded < self.nb_of_files * 0.05:
            self.nb_files_loaded += 1
            self.total_size_seen += data_size
        else:
            if self.probability_cache is None:
                dataset_size = self.nb_of_files * (self.total_size_seen / self.nb_files_loaded)
                self.probability_cache = self.max_size_GB / dataset_size

            if self.current_cache_size < self.max_size_GB:
                if random() < self.probability_cache:
                    self.cached_data[file_path] = loaded_data
                    self.current_cache_size += data_size
