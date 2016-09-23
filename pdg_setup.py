import numpy
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
# from Cython.Build import cythonize

ext_modules = [Extension(
    name="pairwise_distance_gpu",
    sources=["pairwise_distance_gpu.pyx", "pairwise_distance.cpp"],
        # extra_objects=["fc.o"],  # if you compile fc.cpp separately
    include_dirs = [numpy.get_include(), '/usr/local/cuda-7.5/include/'],  # .../site-packages/numpy/core/include
    language="c++",
        # libraries=
        # extra_compile_args = "...".split(),
    extra_link_args = "-L/usr/local/cuda-7.5/lib64/ -lcublas -L/usr/local/cuda-7.5/lib64/ -lcudart".split()
    )]

setup(
    name = 'pairwise_distance_gpu',
    cmdclass = {'build_ext': build_ext},
    ext_modules = ext_modules,
        # ext_modules = cythonize(ext_modules)  ? not in 0.14.1
    # version=
    # description=
    # author=
    # author_email=
    )
