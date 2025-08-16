from linear_reg.src.first_principle import LinearRegression
import unittest


class TestLinearRegression(unittest.TestCase):

    def setUp(self):
        self.model = LinearRegression(batch_size=5, num_features=2)
        self.features_batch = [[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0], [5.0, 6.0]]
        self.labels = [3.0, 5.0, 7.0, 9.0, 11.0]

    def test_get_weights_before_training(self):
        model_weights = self.model.weights
        self.assertEqual(len(model_weights), 2)

    def test_forward_pass(self):
        predictions = self.model.forward(self.features_batch)
        self.assertEqual(len(predictions), 5)
        self.assertEqual(predictions[0], predictions[1])

    def test_successful_back_prop(self):
        predictions = self.model.forward(self.features_batch)
        # weights before back prop
        print(f"Weights before back propagation: {self.model.weights}")
        print(f"Prediction before back propagation: {self.model.forward(self.features_batch)}")
        self.model.back_propagation(features=self.features_batch, predictions=predictions, labels=self.labels, learning_rate=0.005)
        print(f"Weights after back propagation: {self.model.weights}")
        print(f"Prediction after back propagation: {self.model.forward(self.features_batch)}")
    
    def test_training_loop(self):
        predictions = [round(_,4) for _ in self.model.forward(self.features_batch)]
        print(f"Before training.\nPrediction: {predictions}")
        print("Starting training.")
        for index in range(200):
            pred, loss = self.model.train(features=self.features_batch, labels=self.labels, learning_rate=0.0005)
            pred = [round(_,4) for _ in pred]
            print(f'Step {index}. Predictions: {pred}. Loss: {loss:5f}')
            

if __name__ == "__main__":
    unittest.main()
