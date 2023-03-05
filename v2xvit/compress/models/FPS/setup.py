from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='pointnet2_batch_cuda',
    ext_modules=[
        CUDAExtension('pointnet2_batch_cuda', [
            "/".join(__file__.split('/')[:-1] + ['src/pointnet2_api.cpp']),
            "/".join(__file__.split('/')[:-1] + ['src/ball_query.cpp']),
            "/".join(__file__.split('/')[:-1] + ['src/ball_query_gpu.cu']),
            "/".join(__file__.split('/')[:-1] + ['src/group_points.cpp']),
            "/".join(__file__.split('/')[:-1] + ['src/group_points_gpu.cu']),
            "/".join(__file__.split('/')[:-1] + ['src/interpolate.cpp']),
            "/".join(__file__.split('/')[:-1] + ['src/interpolate_gpu.cu']),
            "/".join(__file__.split('/')[:-1] + ['src/sampling.cpp']),
            "/".join(__file__.split('/')[:-1] + ['src/sampling_gpu.cu']),
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })