import setuptools
from pathlib import Path

with open("README.md", "r") as fh:
  long_description = fh.read()

setuptools.setup(
  name="do_bwd",
  version="0.0.1",
  author="Prateek Gundannavar Vijay",
  author_email="prateek@calicolabs.com",
  description="Implementation of Body Weight Dynamics in Diversity Outbred Mice",
  long_description=long_description,
  long_description_content_type="text/markdown",
  url="https://github.com/calico/do_bwd/",
  install_requires=[
    l.strip() for l in Path('requirements.txt').read_text('utf-8').splitlines()
  ],
  packages=setuptools.find_packages(),
  classifiers=[
    "Programming Language :: Python :: 3",
    "License :: OSI Approved :: MIT License",
    "Operating System :: OS Independent",
    "Intended Audience :: Science/Research",
    "Topic :: Genetics",
  ],
  python_requires='>=3.8',
)