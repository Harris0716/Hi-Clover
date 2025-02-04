from setuptools import setup, find_packages

setup(name="HiSiNet", 
version='0.0', 
author="Ediem Al-jibury",
author_email="ealjibur@ic.ac.uk",
description="HiSiNet: a tool for Hi-C analysis",
packages = find_packages(),
install_requires=['cooler==0.8.10', 'frozendict==1.2','scipy>=1.5.2', 'torch>=1.6.0','numpy>=1.18.0', 'Cython==0.29.21', 
                 'hic-straw @ git+https://github.com/aidenlab/straw.git@v0.0.8' ],
python_requires='>=3.6'
)
