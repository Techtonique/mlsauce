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
    
    ext_modules += [Extension("stumpc", 
                ["stumpc.pyx"],
                libraries=["m"],
                #extra_compile_args=["-ffast-math", "-fopenmp"],
                #extra_link_args=["-fopenmp"],
                extra_compile_args=["-ffast-math"],
                include_dirs=[numpy.get_include()])]
    cmdclass.update({'build_ext': build_ext})
    
else:
    
    ext_modules += [Extension("stumpc", 
                ["stumpc.c"],
                libraries=["m"],
                #extra_compile_args=["-ffast-math", "-fopenmp"],
                #extra_link_args=["-fopenmp"],
                extra_compile_args=["-ffast-math"],                
                include_dirs=[numpy.get_include()])]

    
setup(name="stumpc",
      cmdclass=cmdclass,
      ext_modules=ext_modules)
