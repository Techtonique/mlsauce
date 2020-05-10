import numpy
from distutils.core import setup
from distutils.extension import Extension

try:
    from Cython.Distutils import build_ext
except ImportError:
    use_cython = False
else:
    use_cython = True


cmdclass = {}
ext_modules = []


if use_cython:
    
    ext_modules += [Extension("adaoptc", 
                ["adaoptc.pyx"],
                libraries=["m"],
                extra_compile_args=["-ffast-math"],
                include_dirs=[numpy.get_include()])]
    cmdclass.update({'build_ext': build_ext})
    
else:
    
    ext_modules += [Extension("adaoptc", 
                ["adaoptc.c"],
                libraries=["m"],
                extra_compile_args=["-ffast-math"],
                include_dirs=[numpy.get_include()])]

    
setup(name="adaoptc",
      cmdclass=cmdclass,
      ext_modules=ext_modules)
