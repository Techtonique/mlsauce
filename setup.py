import os
import platform
import shutil
import subprocess
import sys
from packaging import version
from pathlib import Path
from setuptools import Command, Extension, setup
from wheel.bdist_wheel import bdist_wheel as _bdist_wheel

# --- Ensure Cython and NumPy are installed before importing them ---
try:
    import Cython
    import numpy
except ImportError as e:
    print("numpy and Cython are not installed: ", str(e))
    # Try installing with uv pip (faster) first, fall back to pip if not available
    try:
        subprocess.run([sys.executable, "-m", "uv", "pip", "install", "Cython>=3.0.10", "numpy>=2.0.0"], check=True)
    except (subprocess.SubprocessError, FileNotFoundError):
        subprocess.run([sys.executable, "-m", "pip", "install", "Cython>=3.0.10", "numpy>=2.0.0"], check=True)
    # Re-attempt imports
    import Cython
    import numpy

class bdist_wheel(_bdist_wheel):
    def finalize_options(self):
        _bdist_wheel.finalize_options(self)
        self.root_is_pure = False
        
    def get_tag(self):
        python, abi, plat = _bdist_wheel.get_tag(self)
        if plat.startswith('linux'):
            plat = 'manylinux2014_x86_64'
        return python, abi, plat

CYTHON_MIN_VERSION = version.parse("3.0.10")
VERSION = "0.32.0"  # also update in pyproject.toml

class clean(Command):
    user_options = [("all", "a", "")]
    
    def initialize_options(self):
        self.all = True
        self.delete_dirs = []
        self.delete_files = []
        
        for root, dirs, files in os.walk("mlsauce"):
            root = Path(root)
            for d in dirs:
                if d == "__pycache__":
                    self.delete_dirs.append(root / d)
            
            if "__pycache__" in root.name:
                continue
                
            for f in (root / x for x in files):
                ext = f.suffix
                if ext == ".pyc" or ext == ".so":
                    self.delete_files.append(f)
                if ext in (".c", ".cpp"):
                    source_file = f.with_suffix(".pyx")
                    if source_file.exists():
                        self.delete_files.append(f)
        
        build_path = Path("build")
        if build_path.exists():
            self.delete_dirs.append(build_path)
    
    def finalize_options(self):
        pass
    
    def run(self):
        for delete_dir in self.delete_dirs:
            shutil.rmtree(delete_dir)
        for delete_file in self.delete_files:
            delete_file.unlink()

EXTENSIONS = {
    "_adaoptc": {"sources": ["mlsauce/adaopt/_adaoptc.pyx"]},
    "_boosterc": {"sources": ["mlsauce/booster/_boosterc.pyx"]},
    "_lassoc": {"sources": ["mlsauce/lasso/_lassoc.pyx"]},
    "_ridgec": {"sources": ["mlsauce/ridge/_ridgec.pyx"]},
    "_stumpc": {"sources": ["mlsauce/stump/_stumpc.pyx"]},
}

def get_module_from_sources(sources):
    for src_path in map(Path, sources):
        if src_path.suffix == ".pyx":
            return ".".join(src_path.parts[:-1] + (src_path.stem,))
    raise ValueError(f"Could not find module from sources: {sources!r}")

def _check_cython_version():
    message = f"Please install Cython with a version >= {CYTHON_MIN_VERSION}"
    try:
        import Cython
    except ModuleNotFoundError:
        raise ModuleNotFoundError(message)
    
    if version.parse(Cython.__version__) < CYTHON_MIN_VERSION:
        message += f" The current version is {Cython.__version__} in {Cython.__path__}."
        raise ValueError(message)

def cythonize_extensions(extensions):
    _check_cython_version()
    from Cython.Build import cythonize
    
    directives = {
        "language_level": "3",
        "embedsignature": True,
        "boundscheck": False,
        "wraparound": False
    }
    
    macros = [("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")]
    
    for ext in extensions:
        if ext.define_macros is None:
            ext.define_macros = macros
        else:
            ext.define_macros += macros
    
    return cythonize(extensions, compiler_directives=directives)

def get_extensions():
    numpy_includes = [numpy.get_include()]
    extensions = []
    
    for config in EXTENSIONS.values():
        name = get_module_from_sources(config["sources"])
        include_dirs = numpy_includes + config.get("include_dirs", [])
        
        extra_compile_args = []
        if sys.platform == "darwin":
            extra_compile_args.extend(["-stdlib=libc++", "-mmacosx-version-min=10.15"])
            if platform.machine() == "arm64":
                extra_compile_args.extend(["-arch", "arm64"])
            else:
                extra_compile_args.extend(["-arch", "x86_64"])
        elif sys.platform == "win32":
            extra_compile_args.extend(["/EHsc", "/O2"])
        
        ext = Extension(
            name=name,
            sources=config["sources"],
            include_dirs=include_dirs,
            extra_compile_args=extra_compile_args,
            language="c",
        )
        extensions.append(ext)
    
    if "sdist" not in sys.argv and "clean" not in sys.argv:
        extensions = cythonize_extensions(extensions)
    
    return extensions

if __name__ == "__main__":
    setup(
        name="mlsauce",
        version=VERSION,
        description="Miscellaneous Statistical/Machine Learning tools",
        long_description="Miscellaneous Statistical/Machine Learning tools",
        author="T. Moudiki",
        author_email="thierry.moudiki@gmail.com",
        license="BSD",
        classifiers=[
            "Development Status :: 3 - Alpha",
            "Intended Audience :: Developers",
            "Natural Language :: English",
            "License :: OSI Approved :: BSD License",
            "Programming Language :: Python :: 3",
            "Programming Language :: Python :: 3.10",
            "Programming Language :: Python :: 3.11",
            "Programming Language :: Python :: 3.12",
        ],
        platforms=["linux", "macosx", "windows"],
        python_requires=">=3.10",
        install_requires=[
            "numpy>=2.0.0",
            "Cython>=3.0.10",
            "jax>=0.4.0",
            "jaxlib>=0.4.0",
            "joblib>=1.0.0",
            "matplotlib>=3.0.0",
            "nnetsauce>=0.15.0",
            "pandas>=2.0.0",
            "requests>=2.0.0",
            "scikit-learn>=1.4.0",
            "scipy>=1.8.0",
            "tqdm>=4.50.0",
            
        ],
        ext_modules=get_extensions(),
        zip_safe=False,
        cmdclass={"clean": clean, "bdist_wheel": bdist_wheel},
        options={
            "bdist_wheel": {
                "universal": False,
                "plat_name": "manylinux2014_x86_64" if sys.platform == "linux" else None,
            }
        },
    )
