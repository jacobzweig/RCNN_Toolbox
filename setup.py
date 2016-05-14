from setuptools import setup
from setuptools import find_packages


setup(name='RCNN_Toolbox',
      version='0.1',
      description='Recurrent Convolutional Neural Networks for Electrophysiological Decoding',
      author='Jacob Zweig',
      author_email='jacob.zweig@gmail.com',
      url='https://github.com/jacobzweig/RCNN_Toolbox',
      download_url='https://github.com/jacobzweig/RCNN_Toolbox/tarball/master',
      license='MIT',
      install_requires=['theano', 'pyyaml', 'six', 'lasagne'],
      extras_require={
      'h5py': ['h5py'],
      },
      packages=find_packages())
