import os
import setuptools

from setuptools import setup

if os.environ["CONDA_DEFAULT_ENV"] in ["sting", "sting_gpu_l", "sting_gpu_w"]:
    # conda requirements are set using the .yml files in the 
    # conda_envs directory.
    requirements = []
else:
    # TODO:add pip requirements here, and make sure you can install 
    # using only pip later.
    requirements = []

setup(
    name='sting',
    version = '0.0.1',
    packages = setuptools.find_packages(),
    install_requires = requirements,
    entry_points= {
        'console_scripts' : [
            'sting.runner = sting.ui.run:main',
            'sting.viewer = sting.ui.viewer:main',
            'sting.runcmd = sting.liveanalysis.run:main',
            'sting.segtrain = sting.segmentation.train:main',
            'sting.segtest = sting.segmentation.test:main',
            'sting.barcodetrain = sting.regiondetect.train:main',
            'sting.barcodetest = sting.regiondetect.test:main',
        ]
    },
    zip_safe=False,
    url='https://github.com/karempudi/sting',
    license='MIT',
    author='Praneeth Karempudi',
    author_email='praneeth.karempudi@gmail.com',
    description=''
)