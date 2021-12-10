import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="auto-diff-colorful-axolotls",
    version="1.0.0",
    description="A package for automatic differentiation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/cs107-colorful-axolotls/cs107-FinalProject",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.6",
    install_requires=['numpy']
)