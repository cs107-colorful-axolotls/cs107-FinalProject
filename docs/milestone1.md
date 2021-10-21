# Milestone 1

## Introduction

## Background

## How to Use _PackageName_

## Software Organization

#### Directory Structure
```
cs107-FinalProject/
    docs/
    src/
        forward_mode.py
        reverse_mode.py
        utilities.py
        __init__.py
    tests/
```
* The `docs/` subdirectory will contain documentation about using the library.

* The `src/` subdirectory will contain further subdirectories containing source code for automated differentiation.

    * The `forward_mode.py` file will contain source code that implements forward mode automated differentiation.
    * The `reverse_mode.py` file will contain source code that implements reverse mode automated differentiation. (While reverse mode is not part of the minimum requirements, we do see that implementing something extra is required, so we are planning ahead here.)
    * The `utilities.py` file will contain source code that implements any helper functions common to both forward and reverse mode automated differentiation as well as code that helps with visualization and debugging.
    * The `__init.py__` file will contain code to import external dependencies and set up automated differentiation.

* The `tests/` subdirectory will contain tests written to be compatible with `pytest`, so that they can be automatically run and code coverage reports can thus be generated.

#### Modules 

* External Dependencies:
    * `numpy` will be used to conduct many of the numerical calculations and array manipulation operations related to automated differentiation.
    * `scipy` has modules that do numerical differentiation. We will not use this module in our implementation, but it does provide a useful benchmark to verify the correctness of our implementation as well as be a performance benchmark for timing.

* Internal Modules: In each of the subdirectories of `src/` we plan on having an `__init__.py` module that imports necessary external dependencies and modules from elsewhere in the codebase. We will also include modules for elementary functions, such as `sin()`, `cos()`, `exp()`, and `log()` to name a few. We will also be overriding many of the dunder methods, such as `__add__()` and `__eq__()` to name a few. We will have these functions return not only the result of the operation on two functions, but also the result of the operation on their derivatives.

#### Testing, Continuous Integration, and Devops

* As described in the `Directory Struture` section above, the test suite will be contained in the `tests/` subdirectory of the project repository.

* We have already set up `TravisCI` to automatically check if builds and tests pass and `CodeCov` to generate code coverage reports. We will each "own" the code that we write, which means being responsible for writing tests for that code. As a best practice, we will attempt maintain greater than 90% code coverage at all times.

* As a general rule, each milestone will have its own branch on the GitHub repository. When code is being implemented, each person will work off of their own branch, or branches. To merge code into the `main` branch, the group member will open a pull request and solicit code reviews and approvals before merging.

#### Distribution

We are planning on two methods of distributing our code.

* GitHub: Users wishing to use our automated differentiation package can clone this repository and run a setup script to get set up.

* Python Package Index (PyPI): Users will also be able to use `pip install` to download and setup our automated differentiation package. We plan to follow the steps outlined [here](https://packaging.python.org/tutorials/packaging-projects/).

#### Packaging and Frameworks

* We plan to follow the steps outlined in the second bullet point of the `Distribution` section to package our code for PyPI.

* While not exactly a framework, we came across [PyScaffold](https://pyscaffold.org/), which helps with formatting and organizing our repository in a way that will make it easier to be packaged and distributed on PyPI.

#### Other Considerations

This is only a preliminary design. As with all software engineering projects, the design and organization of the software will change as the project evolves.

## Implementation

## Licensing
