# setup.py
from setuptools import setup

setup(name='wvpy',
      version='0.1',
      author='John Mount',
      author_email='jmount@win-vector.com',
      url='https://github.com/WinVector/wvpy',
      packages=['wvpy'],
      install_requires=[
          'statistics',
          'numpy',
          'matplotlib',
          'sklearn',
          'pandas'
      ],
      classifiers=[
        'License :: OSI Approved :: BSD-3-Clause'
      ]
)
