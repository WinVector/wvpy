# setup.py
from distutils.core import setup

setup(name='wvpy',
      version='0.1',
      author='John Mount',
      author_email='jmount@win-vector.com',
      url='https://github.com/WinVector/vtreat',
      packages=['wvpy'],
      install_requires=[
          'statistics',
          'numpy',
          'matplotlib',
          'sklearn',
          'pandas'
      ]
)
