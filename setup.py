#!/usr/bin/env python-sirius

import pkg_resources
from setuptools import find_packages, setup


def get_abs_path(relative):
    return pkg_resources.resource_filename(__name__, relative)


with open(get_abs_path("README.md"), "r") as _f:
    _long_description = _f.read().strip()


with open(get_abs_path("VERSION"), "r") as _f:
    __version__ = _f.read().strip()


_requirements = ['', ]
#with open(get_abs_path("requirements.txt"), "r") as _f:
#    _requirements = _f.read().strip().split("\n")


setup(
    name='idanalysis',
    version=__version__,
    author='lnls-fac',
    description='ID Analysis Package',
    long_description=_long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/lnls-fac/idanalysis',
    download_url='https://github.com/lnls-fac/idanalysis',
    license='GPLv3 License',
    classifiers=[
        'Intended Audience :: Science/Research',
        'Programming Language :: Python',
        'Topic :: Scientific/Engineering'
    ],
    packages=find_packages(),
    install_requires=_requirements,
    python_requires=">=3.6",
    zip_safe=False,
    )
