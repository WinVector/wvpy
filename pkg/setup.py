# setup.py
import setuptools

setuptools.setup(name='wvpy',
      version='0.1',
      author='John Mount',
      author_email='jmount@win-vector.com',
      url='https://github.com/WinVector/wvpy',
      packages=setuptools.find_packages(),
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
