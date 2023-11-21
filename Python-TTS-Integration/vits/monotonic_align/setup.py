

from distutils.core import setup
from Cython.Build import cythonize
import numpy
from Cython.Distutils import Extension

# extensions = [
#     Extension(
#         "monotonic_align.core",
#         ["core.pyx"],
#         language_level="3"  # Set the language level here
#     )
# ]

setup(
    name='monotonic_align',
    package_dir={'monotonic_align': ''},
    ext_modules=cythonize("core.pyx", language_level="3"),
    include_dirs=[numpy.get_include()]
)
