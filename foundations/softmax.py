import numpy as np
from numpy.typing import NDArray


class Solution:

    def softmax(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array of logits
        # Hint: subtract max(z) for numerical stability before computing exp
        # return np.round(your_answer, 4)
        
        # denominator = np.sum(np.exp(z)) - np.exp(np.max(z))
        # numerator = np.exp(z) / np.exp(np.max(z))
        # prob = numerator / denominator

        z = z - np.max(z)
        prob = np.exp(z) / np.sum(np.exp(z))

        return np.round(prob, 4)
