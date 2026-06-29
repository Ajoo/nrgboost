import sys
import platform
from cffi import FFI
from os import path

root_dir = path.dirname(path.realpath(__file__))
pcg_inc_dir = path.join(root_dir, 'pcg-c-0.94', 'include')

with open(path.join(root_dir, "eval.h"), "r") as header:
    header.readline()  # skip include
    CDEF = header.read()

include_dirs = [root_dir, pcg_inc_dir]
library_dirs = []
extra_compile_args = ['-O3']
extra_link_args = []

if sys.platform == 'darwin':
    # Apple Clang has no built-in OpenMP; it needs Homebrew's libomp and a
    # different spelling of the flags than GCC. The libomp dylib is bundled
    # into the wheel by delocate (run automatically by cibuildwheel), so end
    # users do not need Homebrew installed.
    brew_prefix = '/opt/homebrew' if platform.machine() == 'arm64' else '/usr/local'
    libomp = path.join(brew_prefix, 'opt', 'libomp')
    include_dirs.append(path.join(libomp, 'include'))
    library_dirs.append(path.join(libomp, 'lib'))
    extra_compile_args += ['-Xpreprocessor', '-fopenmp']
    extra_link_args += ['-lomp']
else:
    # GCC (Linux / manylinux) ships OpenMP as part of the toolchain.
    extra_compile_args += ['-fopenmp']
    extra_link_args += ['-fopenmp']

ffibuilder = FFI()
ffibuilder.cdef(CDEF)
ffibuilder.set_source(
    '_eval',
    '#include "eval.h"',
    sources=[path.join(root_dir, 'eval.c')],
    include_dirs=include_dirs,
    library_dirs=library_dirs,
    extra_compile_args=extra_compile_args,
    extra_link_args=extra_link_args,
)

if __name__ == "__main__":
    ffibuilder.compile(verbose=True)
