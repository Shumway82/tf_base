## Tensorflow Core Library

The tf_core library is a package that includes basic functionalities that help users to quickly and efficiently train models and try new approaches. It contains basic functionalities for cost functions, normalization, activation functions and various layers. In addition, interfaces for models, as well as an entire pipeline, are provided for training and inferencing structures, which allows the loading of data. An example of this pipeline can be found under ./example.

## Pre-requiremtents
* tensorflow >= 1.8 
* pil 
* numpy 
* opencv

## Installation 
1. Clone the repository
```
$ git clone git@github.com:Shumway82/tf_core.git
```
2. Go to folder
```
$ cd tf_core
```
3. Install with pip3
```
$ pip3 install tfcore
or for editing the repository 
$ pip3 install -e .
```

## Usage-Example
```python
import tfcore 
from tfcore.interfaces.ITraining import *
```

## Licensing
tf_core is released under the MIT License (MIT), see LICENSE.
