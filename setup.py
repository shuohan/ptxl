from setuptools import setup

version = '0.2.1'

with open('README.md', 'r', encoding='utf-8') as fh:
    long_description = fh.read()

setup(name='ptxl',
      version=version,
      description='PyTorch training framework using observer design pattern',
      long_description=long_description,
      long_description_content_type='text/markdown',
      author='Shuo Han',
      author_email='shan50@jhu.edu',
      url='https://github.com/shuohan/ptxl',
      license='GPLv3',
      packages=['ptxl'],
      python_requires='>=3.7.7',
      install_requires=['torch >= 1.6.0',
                        'numpy >= 1.18.5',
                        'tqdm >= 4.46.0',
                        'nibabel >= 3.1.1',
                        'matplotlib >= 3.3.2',
                        'Pillow >= 7.2.0'],
      classifiers=['Programming Language :: Python :: 3',
                   'License :: OSI Approved :: MIT License',
                   'Operating System :: OS Independent']
      )
