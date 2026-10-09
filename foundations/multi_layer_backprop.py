import numpy as np
from typing import List


class Solution:
    def forward_and_backward(self,
                              x: List[float],
                              W1: List[List[float]], b1: List[float],
                              W2: List[List[float]], b2: List[float],
                              y_true: List[float]) -> dict:
        # Architecture: x -> Linear(W1, b1) -> ReLU -> Linear(W2, b2) -> predictions
        # Loss: MSE = mean((predictions - y_true)^2)
        #
        # Return dict with keys:
        #   'loss':  float (MSE loss, rounded to 4 decimals)
        #   'dW1':   2D list (gradient w.r.t. W1, rounded to 4 decimals)
        #   'db1':   1D list (gradient w.r.t. b1, rounded to 4 decimals)
        #   'dW2':   2D list (gradient w.r.t. W2, rounded to 4 decimals)
        #   'db2':   1D list (gradient w.r.t. b2, rounded to 4 decimals)
        
        ## Cleaning
        x = np.array(x)                         ## x.shape(1,2)
        W1, b1 = np.array(W1), np.array(b1)     ## W1.shape(2,2)
        W2, b2 = np.array(W2), np.array(b2)     ## W2.shape(1,2)
        print(f"x{x.shape}, w1{W1.shape}, b1{b1.shape}, w2{W2.shape}, b2{b2.shape}")

        ## Forward
        z1 = W1 @ x + b1         ## shape(2,)
        a1 = np.maximum(0, z1)            ## shape(2,)
        y_pred = W2 @ a1 + b2    ## shape(1,)

        ## Loss
        n = len(y_pred)
        loss = np.round(np.square(y_pred - y_true), 4).item()

        ## Backward
        ## dL_dy = 2/n * (y_pred - y_true)
        ## dy_da1 = W2
        ## da1_dz1 = 0 if z1<0 else 1

        ## Activation
        da1_dz1 = np.where(z1>0, 1, 0)
        y_diff = y_pred - y_true

        ## 2nd layer
        dW2 = np.round(2/n * y_diff * a1, 4)
        db2 = np.round(2/n * y_diff, 4)

        ## 1st layer intermediates
        da1 = np.outer((2/n * y_diff), W2)
        dz = da1 * da1_dz1

        print(y_diff.shape, W2.shape, da1_dz1.shape, x.shape)
        dW1 = np.round(np.outer(dz, x), 4)
        db1 = np.round(dz, 4)

        dW1 = dW1.tolist()
        db1 = db1.flatten().tolist()
        dW2 = dW2.reshape(1, -1).tolist()
        db2 = db2.tolist()

        return {"loss":loss, "dW1":dW1, "db1":db1, "dW2":dW2, "db2":db2}
