from setuptools import setup, find_packages

setup(
    name='gelpy',
    version='0.0.1',
    packages=find_packages(),
    author="Christoph Karfusehr",
    author_email="c.karfusehr@gmail.com",
    description="A small python package used for gel electrophoresis visualization, fitting and analysis",
    install_requires=[
        "numpy>=1.24.3",
        "scipy>=1.10.1",
        "scikit-image>=0.21.0",
        "scikit-learn>=1.2.2",
        "matplotlib>=3.7.1",
        "seaborn>=0.12.2",
        "ipykernel>=6.19.2",
        "pandas>=2.0.2",
    ],
)