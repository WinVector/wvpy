# setup.py
import setuptools

DESCRIPTION = "Convert Jupyter notebooks to and from Python files, and render."
LONG_DESCRIPTION = """
Convert Jupyter notebooks to and from Python files, and render.
"""

setuptools.setup(
    name="wvpy",
    version="1.1.2",
    author="John Mount",
    author_email="jmount@win-vector.com",
    url="https://github.com/WinVector/wvpy",
    packages=setuptools.find_packages(exclude=['tests', 'Examples']),
    install_requires=[
        "IPython",
        "nbformat",
        "nbconvert"
    ],
    extras_require = {
        'code_format': ["black"]
    },
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
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: BSD License",
    ],
    long_description=LONG_DESCRIPTION,
)
