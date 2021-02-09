from setuptools import setup

version = '0.2.0'

with open('README.md', 'r', encoding='utf-8') as fh:
    long_description = fh.read()

setup(name='ptxl',
      version=version,
      description='PyTorch training framework using observer design pattern',
      long_description=long_description,
      long_description_content_type='text/markdown',
      author='Shuo Han',
      author_email='shan50@jhu.edu',
      license='GPLv3',
      install_requires=['torch>=1.6.0'],
      packages=['ptxl'],
      classifiers=['Programming Language :: Python :: 3',
                   'License :: OSI Approved :: MIT License',
                   'Operating System :: OS Independent']
      )
