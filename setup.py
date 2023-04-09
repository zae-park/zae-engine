from setuptools import setup, find_packages

__version__ = '0.0.1'

with open('requirements.txt', 'r', encoding='utf-8') as f:
    required = f.read().splitlines()

setup(
    name='zae-engine',
    version=__version__,
    author='zae-park',
    url='https://github.com/zae-park/zae-engine',
    description='Providing scripts for deep learning frameworks',
    long_description='',
    package_data={
        'zae-engine': ['models/resource/.env', 'data/resource/sample_data.*'],
       },
    include_package_data=True,
    packages=find_packages(exclude=["*.tests", "*.tests.*", "tests.*", "tests", 'template.py']),
    install_requires=required,
    zip_safe=False,
)
