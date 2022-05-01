from setuptools import setup

setup(
        name='nmfx',
        version='0.1',
        description='Stochastic Non-Negative Matrix Factorization',
        url='https://github.com/vrutten/nmfx',
        author='Virginia MS Rutten, Jan Funke',
        author_email='ruttenv@janelia.hhmi.org, funkej@janelia.hhmi.org',
        license='MIT',
        packages=[
            'nmfx'
        ],
        install_requires=[
            'numpy',
            'jax',
            'tqdm'
        ],
        scripts=[
            'scripts/nmfx',
            'scripts/nmfx_hdf5'
        ]
)