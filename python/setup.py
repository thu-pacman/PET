import os
import sys
import sysconfig
from setuptools import find_packages
from setuptools import setup
from setuptools.extension import Extension


def config_cython():
    sys_cflags = sysconfig.get_config_var('CFLAGS')
    try:
        from Cython.Build import cythonize
        ret = []
        path = 'tpm/_cython'
        for fn in os.listdir(path):
            if fn.endswith('.pyx'):
                ret.append(Extension(
                    'tpm.%s' % fn[:-4],
                    ['%s/%s' % (path, fn)],
                    include_dirs=['../include'],
                    # libraries=['tpm_runtime'],
                    libraries=[],
                    extra_compile_args=['-DUSE_CUDNN', '-std=c++11'],
                    extra_link_args=[],
                    language='c++',
                ))
        return cythonize(ret, compiler_directives={'language_level': 3})
    except ImportError:
        print("Cython is not installed")
        return []


setup(
    name='tpm',
    version='0.1',
    description='Optimizing Deep Learning Computations with Tensor Mutations',
    zip_safe=False,
    install_requires=[],
    packages=find_packages(),
    url='https://github.com/whjthu/ml-opt',
    ext_modules=config_cython(),
)
