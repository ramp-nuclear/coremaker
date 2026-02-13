from pathlib import Path

from setuptools import find_packages

from conda_setup import setup

dirpath = Path(__file__).parent

with (dirpath / 'README.md').open('r') as f:
    long_description = f.read()

if __name__ == '__main__':
    setup(name='coremaker',
          description="Package for creating cores and core objects",
          long_description=long_description,
          long_description_content_type="text/markdown",
          packages=find_packages(),
          scripts=[],
          entry_points={},
          classifiers=[
              "Programming Language :: Python :: 3",
              "Operating System :: OS Independent",
              ],
          package_data={'': ['*.csv']},
          python_requires='>=3.8',
          requirements_yml='requirements.yml',
          )
