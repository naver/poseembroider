##############################################################
## poseembroider                                            ##
## Copyright (c) 2024                                       ##
## Institut de Robotica i Informatica Industrial, CSIC-UPC  ##
## and Naver Corporation                                    ##
## Licensed under the CC BY-NC-SA 4.0 license.              ##
## See project root for license details.                    ##
##############################################################

from setuptools import setup, find_packages

setup(name='poseembroider',
      version='1.0',
      packages=find_packages('src'),
      package_dir={'': 'src'},
      include_package_data=True,
      author='Ginger Delmas',
      author_email='ginger.delmas.pro@gmail.com',
      description='PoseEmbroider ECCV 2024.',
      long_description=open("README.md").read(),
      long_description_content_type="text/markdown",
      install_requires=[],
      dependency_links=[],
      )