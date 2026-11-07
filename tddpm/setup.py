from setuptools import setup
from Cython.Build import cythonize
import numpy

setup(
    name="tddm",
    ext_modules=cythonize("*.pyx", include_path=[numpy.get_include()], annotate=True),
)