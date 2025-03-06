from setuptools import setup
from torch.utils import cpp_extension

setup(
    name='milo',
    ext_modules=[
        cpp_extension.CUDAExtension(
            'milo_cuda', 
            [
                'milo/milo_cuda.cpp',
                'milo/milo_cuda_kernel.cu',
                'milo/milo_cuda_with_zero_kernel.cu'
            ],
            extra_compile_args={
                'cxx': ['-g', '-O0', '-D_GLIBCXX_USE_CXX11_ABI=0'],
                'nvcc': [
                    '-g',
                    '-O0',
                    '-lineinfo',
                    '-arch=sm_80',
                    '-Xcompiler', '-D_GLIBCXX_USE_CXX11_ABI=0',
                    '--compiler-options', '-fPIC'
                ]
            }
        )
    ],
    cmdclass={'build_ext': cpp_extension.BuildExtension},
    packages=['milo'],
    install_requires=['torch']
)