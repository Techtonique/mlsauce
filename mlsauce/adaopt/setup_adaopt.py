from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

import numpy

ext = Extension("adaoptc", 
                ["adaopt.pyx"],
                libraries=["m"],
                extra_compile_args=["-ffast-math"],
                include_dirs=[numpy.get_include()])

setup(ext_modules=[ext],
      cmdclass = {'build_ext': build_ext})

#from distutils.core import setup
#from Cython.Build import cythonize
#
#setup(name="adaopt", 
#      ext_modules=cythonize("adaopt.pyx"))