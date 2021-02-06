from setuptools import setup

version = '0.1.1'

setup(name='pytorch-trainer',
      version=version,
      description='PyTorch trainer',
      author='Shuo Han',
      author_email='shan50@jhu.edu',
      scripts=scripts,
      install_requires=['torch>=1.6.0'],
      packages=['pytorch_trainer'])
