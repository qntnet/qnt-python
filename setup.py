from setuptools import setup

setup(
    name="qnt",
    version="0.0.119",
    url="https://quantnet.ai",
    license='MIT',
    packages=['qnt', 'qnt.ta'],
    install_requires=['xarray', 'pandas', 'numpy', 'scipy', 'tabulate', 'bottleneck', 'numba']
)