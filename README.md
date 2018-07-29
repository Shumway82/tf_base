## tf_core
# Tensorflow Core Library

The tf_core library is a package that includes basic functionalities that help users to quickly and efficiently train models and try new approaches. It contains basic functionalities for cost functions, normalization, activation functions and various layers. In addition, interfaces for models, as well as an entire pipeline, are provided for training and inferencing structures, which allows the loading of data. An example of this pipeline can be found under ./example.

# Pre-requiremtents
tensorflow >= 1.8 \
pil \
numpy \
opencv

# Installation 
cd ../tf_core \
pip install tfcore \
or \
pip install -e .

# Usage-Example
import tfcore \
or \
import tfcore as tfc \
or \
from tfcore.interfaces.ITraining import *
