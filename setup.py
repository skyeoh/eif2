from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

import numpy
import sys

extra_compile_args = ['-Wcpp']
extra_link_args = []
define_macros = []

# Default mode
_ENABLE_OPENMP = False

for argv in sys.argv:
    # Enable OpenMP
    if argv == '--enable-openmp':
        _ENABLE_OPENMP = True
        sys.argv.remove('--enable-openmp')

if _ENABLE_OPENMP:
    extra_compile_args += ['-fopenmp']
    define_macros += [('ENABLE_OPENMP', '1')]
    extra_link_args += ['-fopenmp']
else:
    define_macros += [('ENABLE_OPENMP', '0')]

setup(
        cmdclass = {'build_ext': build_ext},
        ext_modules = [Extension("eifcxx",
                                  sources=["_eif.pyx", "eif.cxx"],
                                  include_dirs=[numpy.get_include()],
                                  extra_compile_args=extra_compile_args,
                                  extra_link_args=extra_link_args,
                                  define_macros=define_macros,
                                  language="c++")],
)
