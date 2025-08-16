"""Linear regression that leverages the micrograd engine."""

import argparse
from typing import List
from micrograd import Value, AdagradOptimizer, SGDOptimizer
import random

def mse_loss(predictions: List[Value], labels: List[Value]) -> Value:
    """Returns the mean squared loss."""
    return sum((pred - label)**2 for pred, label in zip(predictions, labels)) * (1/len(labels))

class LinearRegression:
    def __init__(self, num_features: int) -> None:
        self.num_features = num_features
        # TODO (addadda023): Add support for other initializers.
        self.weights = [Value(random.uniform(-1, 1)) for _ in range(num_features)]
        self.bias = Value(random.uniform(-1, 1))

    def parameters(self):
        return self.weights + [self.bias]

    def forward(self, features: List[float]) -> Value:
        if len(features) != self.num_features:
            raise ValueError(f"Number of features in input ({len(features)}) does not match model's num_features ({self.num_features})")
        
        # Dot product of weights and features, plus bias
        out = sum([self.weights[i] * features[i] for i in range(self.num_features)], self.bias)
        return out

def train(model: LinearRegression, optimizer, features: List[List[float]], labels: List[float], epochs: int, batch_size: int):
    """Trains the linear regression model using minibatch gradient descent."""
    num_samples = len(features)
    for epoch in range(epochs):
        # Shuffle data
        combined = list(zip(features, labels))
        random.shuffle(combined)
        features, labels = zip(*combined)

        total_loss = 0
        for i in range(0, num_samples, batch_size):
            # Get minibatch
            batch_features = features[i:i+batch_size]
            batch_labels = [Value(l) for l in labels[i:i+batch_size]]

            # Forward pass
            predictions = [model.forward(f) for f in batch_features]
            
            # Compute loss
            loss = mse_loss(predictions, batch_labels)
            total_loss += loss.value

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Update weights
            optimizer.step()

        if (epoch + 1) % 5 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Total Loss per epoch: {total_loss:.4f}, Loss per sample: {total_loss/num_samples:.4f}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a linear regression model.')
    parser.add_argument('--optimizer', type=str, default='sgd', choices=['sgd', 'adagrad'], help='The optimizer to use.')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='The learning rate.')
    parser.add_argument('--epochs', type=int, default=30, help='The number of epochs.')
    parser.add_argument('--batch_size', type=int, default=2, help='The batch size.')
    args = parser.parse_args()

    # Generate some sample data
    X = [[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0], [5.0, 6.0]]
    y = [3.0, 5.0, 7.0, 9.0, 11.0]

    # Initialize the model
    model = LinearRegression(num_features=2)
    
    # Initialize the optimizer
    if args.optimizer == 'sgd':
        optimizer = SGDOptimizer(model.parameters(), learning_rate=args.learning_rate)
    elif args.optimizer == 'adagrad':
        optimizer = AdagradOptimizer(model.parameters(), learning_rate=args.learning_rate)

    # Train the model
    train(model, optimizer, X, y, epochs=args.epochs, batch_size=args.batch_size)
    # Make a prediction
    test_features = [5.0, 6.0]
    prediction = model.forward(test_features)
    print(f"Prediction for {test_features}: {prediction.value}")
