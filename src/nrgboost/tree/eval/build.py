from cffi import FFI
from os import path

root_dir = path.dirname(path.realpath(__file__))
pcg_inc_dir = path.join(root_dir, 'pcg-c-0.94', 'include')
pcg_lib_dir = path.join(root_dir, 'pcg-c-0.94', 'src')

with open(path.join(root_dir, "eval.h"), "r") as header:
    header.readline()  # skip include
    CDEF = header.read()

ffibuilder = FFI()
ffibuilder.cdef(CDEF)
ffibuilder.set_source(
    '_eval',
    '#include "eval.h"',
    sources=[path.join(root_dir, 'eval.c')],
    include_dirs=[root_dir, pcg_inc_dir],
    extra_compile_args=['-fopenmp', '-O3'],
    extra_link_args=['-fopenmp'],
    # libraries=['pcg_random'],
    # library_dirs=[pcg_lib_dir]
)

if __name__ == "__main__":
    ffibuilder.compile(verbose=True)
