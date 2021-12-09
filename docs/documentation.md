# Documentation 

## Introduction
Differentiation is one of the most important operations in mathematics and computation. A derivative measures the sensitivity of change of the output of a function with respect to changes in its input. Derivatives can be calculated symbolically using calculus rules such as the chain rule, product rule, and quotient rule. Symbolically evaluating the derivative ensures accuracy, but these calculations can be costly and might not work for certain functions. We can use numerical methods like the Finite-Difference method to overcome these problems. However, due to rounding errors and possible numerical instability, numerical methods also have their shortcomings. Automatic Differentiation (AD) is less costly than symbolic differentiation, and unlike numerical methods, evaluates derivatives to machine precision. Thus, AD is a good choice for most software that requires differentiation.


## Background
The basic idea of AD is that we can represent complicated functions as a sequence of elementary functions and arithmetic operations. Thus, we can solve complicated derivatives by evaluating their simple components in a step by step manner. AD relies on the chain rule, which gives us a way to calculate the derivative of a compound function. To organize the flow of AD we introduce a graph structure.
Consider the simple one variable function ![equation](https://latex.codecogs.com/png.latex?f=\sin{(x^2)}+x). We can construct the following graph structure to represent the steps needed to evaluate this derivative.


<img src="figures/m1_1.png"/>

Here, each intermediate result is a node in the graph. The nodes are linked together either by arithmetic operations or by elementary functions.
If we want to evaluate the derivative of the function, we can just work through each node in the graph. Suppose we wanted to evaluate the derivative of ![equation](https://latex.codecogs.com/png.latex?f=\sin{(x^2)}+x) at 1, we can construct the following table to calculate the derivative step by step.

<img src="figures/m1_2.png"/>

Note that the value highlighted in yellow is exactly ![equation](https://latex.codecogs.com/png.latex?2x\cos{(x^2)}+1) for ![equation](https://latex.codecogs.com/png.latex?x=1) (what we would get if we evaluated this derivative symbolically. This method of automatic differentiation is known as Forward Mode Automatic Differentiation and can be extended to higher dimensions.

## How to Use Our Package
### How to install 
### Demo

## Organization 

## Implementaiton Details 

## Extension 

## Broader Impact and Inclusivity Statement

Automatic Differentiation is a powerful tool that automates the calculation of complex derivatives. Our package provides a fast, accurate way to calculate the gradients of functions of many variables.  Using our library correctly reduces the risk of human errors. However, it is important that the users of this software read the documentation before use. Misuse of our package could introduce errors into the users’ calculations, potentially invaliding their results. It is a well-documented phenomenon in human factors engineering that humans are less skeptical of results when they come from a machine automated algorithm. In risk adverse fields it is important that the users of this package validate their results to make sure they are using it correctly. 

Our automatic differentiation library is easily accessible to anyone who has access to PyPI. Hosting code on PyPI makes it easy to install with a single line, thus removing barriers to use. We aim to make the library even more user friendly by providing adequate documentation. This documentation explains how to use the package, and the relevant mathematical concepts in a clear, concise manner. Thus, you do not need to be a Python expert or a mathematician to use our library. Those wishing to contribute can request access and submit a pull request to our GitHub repo.


## Future