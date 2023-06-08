from setuptools import setup, find_packages

setup(
    name='gelpy',
    version='0.1',
    packages=find_packages(),
    author="Christoph Karfusehr",
    author_email="c.kafusehr@tum.de",
    install_requires=[
        "matplotlib",
        "numpy",
        "scipy",
        "scikit-image",
        "pandas",
        "seaborn",
        "Ipython",
        "ipykernel",
    ],
)