"""Linear Regression from first principles without any autograd library."""

from typing import List
from random import random


def dot_product(a_vector, b_vector):
    """Calculates the dot product between two lists."""
    return sum([an*bn for an,bn in zip(a_vector,b_vector)])

def mse_loss(predictions, labels):
    """Calculates the Mean Squared Loss."""
    return sum([(predictions[index] - labels[index])**2 for index in range(len(predictions))])/len(predictions)


class LinearRegression():
    """Creates a linear regression model.

    Supports forward propagation, back propagation, and 
    prediction.

    y = a1 * x1 + a2 * x2 + .. + a_n * x_n + b
    """
    def __init__(self, batch_size: int, num_features: int) -> None:
        self.batch_size = batch_size
        self.num_features = num_features
        # TODO (addadda023): Add utility to initialize with other methods.
        # self.weights = [random() for index in range(num_features)]
        self.weights = [0 for index in range(num_features)]
        self.bias = random()
        self.optimizer_steps = 0
        self.losses = []
        self.batch_loss = []
        
    def forward(self, features: List[List[float]]) -> List[float]:
        """Performs forward propagation."""
        return [dot_product(f, self.weights) + self.bias for f in features]
        
    
    def back_propagation(self, features:List[List[float]], predictions: List[float], labels: List[float], learning_rate:float) -> None:
        """Back propagation from first principle."""
        losses = [p - l for p, l in zip(predictions, labels)]
        self.losses.append(losses)

        # Calculate gradients for weights.
        gradients = []
        for index in range(self.num_features):
            gradient = sum([features[row][index] * losses[row] for row in range(self.batch_size)])
            gradients.append(gradient * 2 / self.batch_size)
        
        # Calculate Gradient for bias
        bias_gradient = 2 / self.batch_size * sum(losses)

        # Update the weights and bias
        for index in range(self.num_features):
            self.weights[index] -= learning_rate * gradients[index]
        self.bias -= learning_rate * bias_gradient

        # Calculate the batch loss.
        loss = mse_loss(predictions=predictions, labels=labels)
        self.batch_loss.append(loss)
        
        # Update the optimizer step
        self.optimizer_steps += 1

        return loss

    
    def train(self, features: List[List[float]], labels: List[float], learning_rate=0.005) -> None:
        if len(features) != self.batch_size:
            raise ValueError(f"The provided batch_size {len(features)} doesn't match with model batch_size {self.batch_size}.")
        predictions = self.forward(features=features)
        batch_loss = self.back_propagation(features=features, predictions=predictions, labels=labels, learning_rate=learning_rate)
        return predictions, batch_loss
        # TODO (addadda023): Add utilities to save the batch, predictions and loss to disk.
