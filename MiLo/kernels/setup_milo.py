# from setuptools import setup
# from torch.utils import cpp_extension

# setup(
#     name='milo',
#     version='0.1.1',
#     author='Elias Frantar',
#     author_email='elias.frantar@ist.ac.at',
#     description='Highly optimized FP16xINT4 CUDA matmul kernel.',
#     install_requires=['numpy', 'torch'],
#     packages=['milo'],
#     ext_modules=[cpp_extension.CUDAExtension(
#         'milo_cuda', ['milo/milo_cuda.cpp', 'milo/milo_cuda_kernel.cu','milo/milo_cuda_with_zero_kernel.cu'],
#         extra_compile_args={
#     'gcc': ['-g', '-O0'],
#     'nvcc': ['-g', '-O0', '-lineinfo', '-arch=sm_80']  #-G 和 -lineinfo不能一起使用； -G优先级更高
#     },
#     extra_link_args=['-lcudart']
#     )],
#     cmdclass={'build_ext': cpp_extension.BuildExtension},
# )

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