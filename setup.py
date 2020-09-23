from setuptools import setup

setup(
    name="qnt",
    version="0.0.144",
    url="https://quantnet.ai",
    license='MIT',
    packages=['qnt', 'qnt.ta', 'qnt.data'],
    install_requires=['xarray', 'pandas', 'numpy', 'scipy', 'tabulate', 'bottleneck', 'numba']
)