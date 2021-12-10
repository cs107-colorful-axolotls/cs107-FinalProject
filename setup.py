from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="adfihenryi",
    version="1.0.0",
    description="A package for forward and reverse automatic differentiation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/cs107-colorful-axolotls/cs107-FinalProject",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
