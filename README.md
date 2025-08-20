# Two implementations of Linear Regression

## From first principle
This approach calculates the gradients by taking the literal derivative of the loss with respect to each weight. 

## Using micrograd engine
This approach leverages micrograd to propagate the gradients using chain rule. The following Optimizers are implemented:
* SGD
* Adagrad
* Momentum
* RMSProp
* Adam

