import os, sys
from setuptools import setup, find_packages
PACKAGE_NAME = 'sample_pakage_model'


def read_requirements():
    """Parse requirements from requirements.txt."""
    reqs_path = os.path.join('.', 'requirements.txt')
    with open(reqs_path, 'r') as f:
        requirements = [line.rstrip() for line in f]
    return requirements


setup(
     name=PACKAGE_NAME,
      vesion='1.0.1',
      description='image classification',
      packages=find_packages(),
      include_package_data=True,
      install_requires=read_requirements(),
      )