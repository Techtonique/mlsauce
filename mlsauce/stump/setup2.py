import os 
from distutils.core import setup
from Cython.Build import cythonize

dir_path = os.path.dirname(os.path.realpath(__file__))

setup(ext_modules=cythonize(os.path.join(dir_path, "_stumpc.pyx"), 
                            compiler_directives={'language_level' : "3"}))