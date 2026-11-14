# import functions from linreg.py
from foundations.linreg_multi import make_data, train

import numpy as np

def test_learns_true_params():
    X,y = make_data()
    # more epochs here
    w,b, _ = train(X,y, lr=0.01, epochs=5000)

    #assert = this must be true if nore the test fails
    #abs(w-3)<0.2 means the w is withing 0.2 of the true value 3
    assert np.allclose(w, true_w, atol=0.2) # check if all weights are close to true weights
    assert abs(b-2)<0.3 # b is a bit noiser to learn therefore looser tolerance