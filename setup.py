#
# Created by mae9785 (eremeev@nyu.edu)
#

from setuptools import setup, find_packages
import setuptools.command.build_py as build_py


setup_kwargs = dict(
    name='elsa',
    version='0.0.2',
    packages=find_packages(),
    install_requires=[
    ],
    setup_requires=[
    ],

    cmdclass={'build_py': build_py.build_py},
)

setup(**setup_kwargs)
