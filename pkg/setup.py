# setup.py
import setuptools

setuptools.setup(name='wvpy',
      version='0.1.1',
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
          'Intended Audience :: Science/Research',
          'Programming Language :: Python',
          'Programming Language :: Python :: 3',
          'Programming Language :: Python :: 3.5',
          'Programming Language :: Python :: 3.6',
          'Programming Language :: Python :: 3.7',
          'License :: OSI Approved :: BSD License',
      ],
)
