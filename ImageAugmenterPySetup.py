import numpy
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
# from Cython.Build import cythonize

ext_modules = [Extension(
    name="ImageAugmenterPy",
    sources=["ImageAugmenterPy.pyx"],
    extra_objects=["img_aug.o"],  # if you compile fc.cpp separately
    include_dirs = [numpy.get_include()],  # .../site-packages/numpy/core/include
    language="c++",
    # libraries=
    # extra_compile_args = "...".split(),
    extra_link_args = " -lopencv_core -lopencv_highgui -lopencv_imgproc".split()
    )]
print(ext_modules[0])    

setup(
    name = 'ImageAugmenterPy',
    cmdclass = {'build_ext': build_ext},
    ext_modules = ext_modules,
        # ext_modules = cythonize(ext_modules)  ? not in 0.14.1
    # version=
    # description=
    # author=
    # author_email=
    )


"""
Test code piece

import cv2
import ImageAugmenterPy

img = cv2.imread('/tmp/test/test.jpg')
args = {'data_shape':(3, 100,100)}
aug_img = ImageAugmenterPy.augment_img_process(img,args=args)
print(aug_img.shape)
cv2.imshow("aug_img", aug_img)
cv2.waitKey()
"""
