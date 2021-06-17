from setuptools import setup

version = '0.3.1'

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
      python_requires='>=3.7.10',
      install_requires=[
          'torch>=1.8.1',
          'numpy',
          'tqdm',
          'nibabel',
          'matplotlib',
          'Pillow'
      ],
      classifiers=[
          'Programming Language :: Python :: 3',
          'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
          'Operating System :: OS Independent']
      )
