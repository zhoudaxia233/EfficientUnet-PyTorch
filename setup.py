import os
import sys
import platform
import efficientunet
from setuptools import setup, find_packages


# "setup.py publish" shortcut.
if sys.argv[-1] == 'publish':
    os.system('python setup.py sdist')
    os.system('twine upload dist/*')
    if platform.system() == 'Windows':
        os.system('powershell rm –path dist, efficientunet_pytorch.egg-info –recurse –force')
    else:
        os.system('rm -rf dist efficientunet_pytorch.egg-info')
    sys.exit()

install_requires = ['torch>=1.0.0']

setup(
    name='efficientunet-pytorch',
    version=efficientunet.__version__,
    description="A PyTorch 1.0 Implementation of Unet with EfficientNet as encoder",
    url='https://github.com/zhoudaxia233/EfficientUnet-PyTorch',
    license='MIT',
    packages=find_packages(),
    include_package_data=True,
    zip_safe=False,
    install_requires=install_requires,
    python_requires='>=3.6'
)
