from setuptools import find_packages, setup

setup(
    name='cml',
    packages=find_packages(exclude=['tests']),
    version='0.1.0',
    description='Библиотека для того, чтобы потыкать классическое машинное обучение.'
)