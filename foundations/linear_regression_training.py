import numpy as np
from numpy.typing import NDArray


class Solution:
    def get_derivative(self, model_prediction: NDArray[np.float64], ground_truth: NDArray[np.float64], N: int, X: NDArray[np.float64], desired_weight: int) -> float:
        # note that N is just len(X)
        return -2 * np.dot(ground_truth - model_prediction, X[:, desired_weight]) / N

    def get_model_prediction(self, X: NDArray[np.float64], weights: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.squeeze(np.matmul(X, weights))

    def update_weights(self, gradient, weights, learning_rate):
        weights -= gradient * learning_rate

        return weights
    

    learning_rate = 0.01

    def train_model(
        self, 
        X: NDArray[np.float64], 
        Y: NDArray[np.float64], 
        num_iterations: int, 
        initial_weights: NDArray[np.float64]
    ) -> NDArray[np.float64]:

        # you will need to call get_derivative() for each weight
        # and update each one separately based on the learning rate!
        # return np.round(your_answer, 5)
        weights = initial_weights
        ground_truth = Y

        for i in range(num_iterations):
            pred_result = self.get_model_prediction(X, weights)
            
            gradient_1 = self.get_derivative(pred_result, ground_truth, len(X), X, desired_weight=0)
            gradient_2 = self.get_derivative(pred_result, ground_truth, len(X), X, desired_weight=1)
            gradient_3 = self.get_derivative(pred_result, ground_truth, len(X), X, desired_weight=2)

            weights[0] = self.update_weights(gradient_1, weights[0], self.learning_rate)
            weights[1] = self.update_weights(gradient_2, weights[1], self.learning_rate)
            weights[2] = self.update_weights(gradient_3, weights[2], self.learning_rate)

        return np.round(weights, 5)











        
