from setuptools import setup, find_packages
from os import path

readme_folder = path.dirname(path.abspath(__file__))
readme_path = path.join(readme_folder, 'README.md')

with open(readme_path) as file:
    long_description = file.read()
    

long_description = '''
# 🔋 NRGBoost: Energy-Based Generative Boosted Trees

Official implementation of the [NRGBoost](https://arxiv.org/abs/2410.03535) algorithm.

Github: https://github.com/ajoo/nrgboost

''' + long_description[long_description.find('## Installation'):]

setup(
    name='nrgboost',
    version='0.0.2',
    description='Official NRGBoost implementation',
    author='João Bravo',
    url='https://github.com/ajoo/nrgboost',
    packages=find_packages(where='src'), #['nrgboost', 'nrgboost.tree', 'nrgboost.tree.eval'],
    package_dir={'': 'src'},
    install_requires=[
        'cffi>=1',
        'numpy',
        'scipy',
        'numba',
        'tqdm',
        'joblib',
        'pandas',
    ],
    setup_requires=['cffi>=1'],
    cffi_modules=['src/nrgboost/tree/eval/build.py:ffibuilder'],
    long_description=long_description,
    long_description_content_type='text/markdown',
)
