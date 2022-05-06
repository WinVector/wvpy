# setup.py
import setuptools

DESCRIPTION = "Simple utilities for teaching Pandas and scikit learn."
LONG_DESCRIPTION = """
Simple utilities for teaching Pandas, Jupyter, seaborn, and sklearn.
"""

setuptools.setup(
    name="wvpy",
    version="0.3.2",
    author="John Mount",
    author_email="jmount@win-vector.com",
    url="https://github.com/WinVector/wvpy",
    packages=setuptools.find_packages(exclude=['tests', 'Examples']),
    install_requires=[
        "numpy",
        "pandas",
        "sklearn",
        "seaborn",
        "matplotlib",
        "vtreat>=1.1.1",
        "data_algebra>=1.2.0",
        "IPython",
        "nbformat",
        "nbconvert"
    ],
    platforms=["any"],
    license="License :: OSI Approved :: BSD 3-clause License",
    description=DESCRIPTION,
    long_description_content_type="text/markdown",
    classifiers=[
        "Intended Audience :: Science/Research",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "License :: OSI Approved :: BSD License",
    ],
    long_description=LONG_DESCRIPTION,
)
