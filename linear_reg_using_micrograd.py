"""Linear regression that leverages the micrograd engine."""

from typing import List
from micrograd import Value
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

    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0

    def forward(self, features: List[float]) -> Value:
        if len(features) != self.num_features:
            raise ValueError(f"Number of features in input ({len(features)}) does not match model's num_features ({self.num_features})")
        
        # Dot product of weights and features, plus bias
        out = sum([self.weights[i] * features[i] for i in range(self.num_features)], self.bias)
        return out

    def update_params(self, learning_rate: float):
        for p in self.parameters():
            p.value -= learning_rate * p.grad

def train(model: LinearRegression, features: List[List[float]], labels: List[float], learning_rate: float, epochs: int, batch_size: int):
    """Trains the linear regression model using minibatch gradient descent."""
    num_samples = len(features)
    for epoch in range(epochs):
        # Shuffle data
        combined = list(zip(features, labels))
        random.shuffle(combined)
        features, labels = zip(*combined)

        total_loss = 0
        for i in range(0, batch_size, num_samples):
            # Get minibatch
            batch_features = features[i:i+batch_size]
            batch_labels = [Value(l) for l in labels[i:i+batch_size]]

            # Forward pass
            predictions = [model.forward(f) for f in batch_features]
            
            # Compute loss
            loss = mse_loss(predictions, batch_labels)
            total_loss += loss.value

            # Backward pass
            model.zero_grad()
            loss.backward()
            
            # Update weights
            model.update_params(learning_rate)

        if (epoch + 1) % 5 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Total Loss per epoch: {total_loss:.4f}, Loss per sample: {total_loss/num_samples:.4f}')


if __name__ == "__main__":
    # Generate some sample data
    X = [[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0], [5.0, 6.0]]
    y = [3.0, 5.0, 7.0, 9.0, 11.0]

    # Initialize the model
    model = LinearRegression(num_features=2)

    # Train the model
    train(model, X, y, learning_rate=0.003, epochs=30, batch_size=2)
    # Make a prediction
    test_features = [5.0, 6.0]
    prediction = model.forward(test_features)
    print(f"Prediction for {test_features}: {prediction.value}")
