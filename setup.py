from setuptools import find_packages
from setuptools import setup
from setuptools.command.install import install

try:
    import tensorflow
except ImportError:
    raise ImportError('Tensorflow is required to install this package.'
                      'Please use `pip install tensorflow` or '
                      '`pip install tensorflow-gpu`')

setup(name='tfcore',
      version='0.1',
      description='Deep learning library',
      url='http://github.com/storborg/funniest',
      author='Silvio Jurk',
      author_email='silvio.jurk@googlemail.com',
      license='MIT',
      install_requires=['markdown',
                        'keras',
                        'numpy',
                        'scikit-learn',
                        'scipy',
                        'matplotlib',
                        'pytest',
                        'scikit-image',
                        'imageio',
                        'pytest',
                        'pydoc-markdown',
                        'flake8'],
      packages=find_packages())
