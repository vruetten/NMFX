from setuptools import setup
from pathlib import Path

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name="nmfx",
    version="0.11",
    description="Stochastic Non-Negative Matrix Factorization",
    url="https://github.com/vrutten/nmfx",
    author="Virginia MS Rutten, Jan Funke",
    author_email="ruttenv@janelia.hhmi.org, funkej@janelia.hhmi.org",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license="MIT",
    packages=["nmfx"],
    install_requires=["numpy", "jax", "tqdm"],
    scripts=["scripts/nmfx", "scripts/nmfx_hdf5"],
)
